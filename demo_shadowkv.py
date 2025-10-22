#%%

import torch

import shadowkv_lib
import json

# shadowkv_lib.Llama

from transformers import AutoTokenizer

# print("Model class:", model_calss)

#%%

model_name = "llama_hf/Meta-Llama-3.1-8B-Instruct"

model = shadowkv_lib.Llama(
    model_name,
)


#%%

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


#%%
# with open("./datasets/tmp_ruler_cwe_128k_0.json", "r") as f:
with open("./datasets/longbench-v2.json", "r") as f:
    data = json.load(f)
    context = data[-1]["context"] # for long bench
    # context = data["0"]["origin_prompt"][0]["prompt"]

token = tokenizer(
    context,
    max_length=56000,
    truncation=True,
    return_tensors=None,
)
context = tokenizer.decode(token.input_ids, skip_special_tokens=True)

inputs = tokenizer.apply_chat_template(
    [
        [
            # {"role": "system", "content": "You are a helpful assistant. You can solve the question quick and efficiently."},
            # {"role": "user", "content": f"key=213213213213213231\n{truncated_str}\n\nSummary of the provided paper in detail."},
            {"role": "user", "content": f"{context}\n\nRefine the given paper, and write into Latex Format."},
            # {"role": "user", "content": context},
        ],
    ],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("number of tokens:", inputs.input_ids.shape[-1])

#%%
with torch.no_grad():
    # Generate text
    output = model.generate(
        input_ids=inputs.input_ids.to(model.device),
        gen_len=4096*2,
        benchmark=True,
        verbose=True,
    )

    # Decode generated text
    # print(output[0])

# %%
