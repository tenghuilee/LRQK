#%%
import kvquant_lib as kvquant
from transformers import AutoTokenizer, TextStreamer
from transformers.tokenization_utils import PreTrainedTokenizer
import json
import torch

import lrqk_attention as lrqk

#%%
# model_name_or_path = "./llama_hf/Meta-Llama-3.1-8B-Instruct"
model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

config = kvquant.KVQuantConfig(
    model_name_or_path=model_name_or_path,
    quantizer_path="./kvquant.pickle/quantizers_Qwen2.5-7B-Instruct.pickle",
)

model = kvquant.create_kvquant_model(config, device="cuda")

print(model)

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # type: PreTrainedTokenizer

#%%

model, tokenizer = lrqk.load_model(model_name_or_path, lrqk=True, base_model=model)

print(model)

#%%
_conf = model.config

if hasattr(_conf, "num_key_value_heads"):
    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
elif hasattr(_conf, "multi_query_group_num"):
    num_key_value_groups = model.config.multi_query_group_num

#%%

with open("./datasets/tmp_ruler_cwe_128k_0.json", "r") as f:
# with open("./datasets/longbench-v2.json", "r") as f:
    data = json.load(f)
    # context = data[-1]["context"] # for long bench
    context = data["0"]["origin_prompt"][0]["prompt"]

# token = tokenizer(context, max_length=40000, truncation=True, return_tensors=None)
token = tokenizer(context, max_length=32000,
                  truncation=True, return_tensors=None)
print("number of tokens:", len(token.input_ids))
context = tokenizer.decode(token.input_ids, skip_special_tokens=True)


# %%

inputs = tokenizer.apply_chat_template(
    [
        [
            # {"role": "system", "content": "You are a helpful assistant. You can solve the question quick and efficiently."},
            # {"role": "user", "content": f"key=213213213213213231\n{truncated_str}\n\nSummary of the provided paper in detail."},
            # {"role": "user", "content": f"key=213213213213213231\n{truncated_str}\n\nList all important points of the provided paper into Markdown format."},
            {"role": "user", "content": context},
            # {"role": "user", "content": questions[1]},
        ],
        # [
        #     {"role": "system", "content": "You are a helpful assistant. You can solve the question quick and efficiently."},
        #     {"role": "user", "content": f"key=213213213213213231\n\n\nWhat is the key?"},
        #     {"role": "assistant", "content": "The key is 213213213213213231."},
        #     {"role": "user", "content": "What is the weather like?"},
        #     {"role": "assistant", "content": "The weather is sunny."},
        # ],
    ],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    # max_length=128000,
    truncation=True,
    padding=True,
    # tokenize=False,
)

stream  = TextStreamer(tokenizer, skip_prompt=True)

#%%

with torch.no_grad():
    outputs = model.generate(
        **inputs.to(model.device),
        max_new_tokens=1024,
        past_key_values=lrqk.DynamicLRQKCache(
            num_key_value_groups=num_key_value_groups,
            # r=32,
            r=32,
            num_active_tokens=2048,
            lite_tokens=64,
            max_iter=(10, 10),
            tol=(1e-8, 1e-8),
            lwattn_factory=lrqk.LightAttentionIndicesOffloadPrefill,
        ),
        streamer=stream,
    )



#%%

