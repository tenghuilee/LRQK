import json
import time

import torch
import transformers
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen2 import modeling_qwen2

from lrqk_attention import *

# TRITON_LIBCUDA_PATH =
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["TRITON_LIBCUDA_PATH"] = "~/Tools/cuda-12.1/targets/x86_64-linux/lib/stubs"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# TRITON_LIBCUDA_PATH = /work1/tenghui/Tools/cuda12.6/targets/x86_64-linux/lib/stubs
# CUDA_HOME = /work1/tenghui/Tools/cuda12.6


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0")
# %%
# force rewrite the attention class

# mistralai/Mistral-7B-Instruct-v0.3
# model_name = "GitLFS/Mistral-7B-Instruct-v0.3"

# microsoft/Phi-3-mini-128k-instruct
# model_name = "GitLFS/Phi-3-mini-128k-instruct"
# model_name = "GitLFS/Yarn-Mistral-7b-128k"
model_name = "./llama_hf/Meta-Llama-3.1-8B-Instruct"
# model_name = "Qwen/Qwen2.5-14B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"

# qwen_yarn = {
#     "max_position_embeddings": 32768 * 4,
#     "rope_scaling": {
#         "factor": 4.0,
#         "original_max_position_embeddings": 32768,
#         "type": "yarn"
#     }
# }

# model, tokenizer = load_lrqk_model(
#     # "Qwen/Qwen2.5-7B-Instruct",
#     model_name,
#     # "GitLFS/Phi-mini-MoE-instruct",
#     # "microsoft/Phi-mini-MoE-instruct",
#     # "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
#     # "./llama_hf_mirror/llama_hf/llama-3-8b-instruct",
#     # "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
#     # "THUDM/glm-4-9b-chat",
#     device=device,
#     # yarn=qwen_yarn,
# )


model, tokenizer = load_model(
    model_name,
    # lrqk=False,
    lrqk=True,
    # yarn=qwen_yarn,
    device=device,
)

# print(model.config)

# %%
# with open("./datasets/tmp_ruler_cwe_128k_0.json", "r") as f:
with open("./outputs/empty/predictions/empty/ruler_cwe_16k.json", "r") as f:
# with open("./datasets/longbench-v2.json", "r") as f:
    data = json.load(f)
    # context = data[-1]["context"] # for long bench
    _data_item = data["3"]
    context = _data_item["origin_prompt"][0]["prompt"]
    print(_data_item["gold"])

# token = tokenizer(context, max_length=40000, truncation=True, return_tensors=None)
# token = tokenizer(
#     context,
#     max_length=56000,
#     # max_length=58000,
#     truncation=True,
#     return_tensors=None,
# )
# context = tokenizer.decode(token.input_ids, skip_special_tokens=True)

inputs = tokenizer.apply_chat_template(
    [
        [
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": f"key=213213213213213231\n{context}\n\nSummary of the provided paper in detail."},
            # {"role": "user", "content": f"{context}\n\nRefine the given paper, and write into Latex Format."},
            {"role": "user", "content": context},
        ],
    ],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("number of tokens:", inputs.input_ids.shape[-1])

# %%

streamer = transformers.TextStreamer(tokenizer, skip_prompt=True)

_conf = model.config

if hasattr(_conf, "num_key_value_heads"):
    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
elif hasattr(_conf, "multi_query_group_num"):
    num_key_value_groups = model.config.multi_query_group_num

# print(model.generation_config)
torch.cuda.empty_cache()
in_seq_len = inputs.input_ids.shape[1]
inputs = inputs.to(model.device)

with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        time_begin = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096*8,
            # past_key_values=cache_utils.DynamicCache(),
            # eos_token_id=tokenizer.eos_token_id,
            past_key_values=DynamicLRQKCache(
                num_key_value_groups=num_key_value_groups,
                r=32,
                num_active_tokens=2048,
                lite_tokens=64,
                max_iter=(2, 2),
                tol=1e-8,
                # lwattn_factory=LightAttentionIndicesOffloadPrefill,
                init_aq_ak_method=InitAQAK.randn,
            ),
            do_sample=True,
            streamer=streamer if inputs.input_ids.shape[0] == 1 else None,
        )
        time_end = time.time()

    print("number of input tokens:", in_seq_len)
    print("number of output tokens:", outputs.shape[-1] - in_seq_len)
    print("time cost:", time_end - time_begin)


# %%

# # histogram of the average time cost
# time_cost_records = []
# for _ in range(20):
#     with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16):
#         time_begin = time.time()
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=128,
#             # past_key_values=cache_utils.DynamicCache(),
#             # eos_token_id=tokenizer.eos_token_id,
#             past_key_values=DynamicLRQKCache(
#                 num_key_value_groups=num_key_value_groups,
#                 r=32,
#                 num_active_tokens=2048,
#                 lite_tokens=64,
#                 max_iter=(2, 2),
#                 tol=(1e-4, 1e-4),
#                 lwattn_factory=LightAttentionIndicesOffloadPrefill,
#             ),
#             do_sample=False,
#         )
#         time_end = time.time()
#     time_cost_records.append(time_end - time_begin)
#     print("time cost:", time_cost_records[-1])

# print("time cost records:", time_cost_records)
# print("average time cost:", sum(time_cost_records) / len(time_cost_records))
