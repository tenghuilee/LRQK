"""
histogram of the lowrankness of the KV heads in Salesforce/wikitext

"""

#%%

import itertools
import json
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from collections import defaultdict

import torch
import transformers
from torch.utils.data import Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, cache_utils)

import lrqk_attention
from tools.wikitext import WikiTextWrapper


#%%
class HitRateCounter(lrqk_attention.LightAttentionIndicesFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.miss_rates = []

    def only_indices_in_cpu(self, now_indices: torch.Tensor, new_indices: torch.Tensor):
        updated_indices, now_in_new = super().only_indices_in_cpu(now_indices, new_indices)

        # now_in_new; if 1: miss, need to load from CPU
        miss_rate = torch.sum(now_in_new) / now_in_new.numel()
        self.miss_rates.append(miss_rate.item())
        return updated_indices, now_in_new
    
    def average_miss_rate(self):
        return np.mean(self.miss_rates)

def get_new_cache(
    model: PreTrainedModel,
    r=16,
    num_active_tokens=256,
    lite_tokens=16,
):
    _conf = model.config
    num_key_value_groups = _conf.num_attention_heads // _conf.num_key_value_heads

    cache = lrqk_attention.DynamicLRQKCache(
        r=r,
        num_active_tokens=num_active_tokens,
        lite_tokens=lite_tokens,
        num_key_value_groups=num_key_value_groups,
    )

    cache.lwattn = defaultdict(lambda: HitRateCounter(
        num_lite_tokens=lite_tokens,
        attn_topk=num_active_tokens,
        num_key_value_groups=num_key_value_groups,
        r=cache.r,
        max_iter=cache.max_iter,
        tol=cache.tol,
        capacity=4096,
    ))
    return cache
    

#%%

def iter_one(
    model_name: str,
    model_abbr: str,
    out_path: str,
    list_r:list[int],
    list_num_active_tokens: list[int],
    list_lite_tokens:list[int],
):
    model, tokenizer = lrqk_attention.load_lrqk_model(
        model_name,
        device="cuda:0",
    )

    dataset = WikiTextWrapper()

    output_data = list()

    for r, na, nl in itertools.product(list_r, list_num_active_tokens, list_lite_tokens):
        print(f"model: {model_abbr}, r: {r}, num_active_tokens: {na}, lite_tokens: {nl}")
        for src in tqdm(dataset):
            inputs = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Summarize the following text."},
                    {"role": "user", "content": src["content"]},
                ],
                add_generation_prompt=True,
                return_tensors="pt",
                # padding=True,
                max_length=4096,
                return_dict=True,
            )

            inputs = inputs.to(model.device)

            with torch.no_grad():
                past_key_values = get_new_cache(model)
                outputs = model.generate(
                    **inputs,
                    past_key_values=past_key_values,
                    max_new_tokens=16,
                    do_sample=True,
                )

                for layer, cache in past_key_values.lwattn.items():
                    output_data.append({
                        "model": model_abbr,
                        "r": r,
                        "num_active_tokens": na,
                        "lite_tokens": nl,
                        "layer": layer,
                        "miss_rate": cache.average_miss_rate(),
                    })
        with open(out_path, "w", encoding="utf-8") as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    return output_data

_model_meta = [
    # {
    #     "model_name": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    #     "model_abbr": "llama-3-8B-1M",
    #     "output_path": "./figoutput/llama-3-8B-1M_hitrate.jsonl",
    # },
    {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_abbr": "qwen-2.5-7B",
        "output_path": "./figoutput/qwen-2.5-7B_hitrate.jsonl",
    }
]

# for item in _model_meta:
#     output_data = iter_one(
#         model_name=item["model_name"],
#         model_abbr=item["model_abbr"],
#         out_path=item["output_path"],
#         list_r=[8, 16, 32, 64],
#         list_num_active_tokens=[128, 256, 512],
#         list_lite_tokens=[4, 8],
#     )
    
#%%

def iter_jsonl(src_paths: list[str]):
    for src in src_paths:
        with open(src, "r") as f:
            for line in f:
                item = json.loads(line)
                # if is nan
                if math.isnan(item["miss_rate"]):
                    continue
                yield item


df_llama3 = pd.DataFrame(iter_jsonl([
    "./figoutput/llama-3-8B-1M_hitrate.jsonl",
    "./figoutput/llama-3-8B-1M_hitrate.jsonl_1024.jsonl",
]))

df_qwen = pd.DataFrame(iter_jsonl([
    "./figoutput/qwen-2.5-7B_hitrate.jsonl",
    "./figoutput/qwen-2.5-7B_hitrate.jsonl_1024.jsonl",
]))

print(df_llama3)
print(df_qwen)

#%%

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
fs = df_llama3["miss_rate"]
plt.hist(fs, bins=100)
plt.xlabel(f"Llama-3-8B-1M,  {fs.mean():.4f}±{fs.std():.4f}")
plt.subplot(1, 2, 2)
fs = df_qwen["miss_rate"]
plt.hist(fs, bins=100)
plt.xlabel(f"Qwen-2.5-7B,  {fs.mean():.4f}±{fs.std():.4f}")
plt.savefig(f"./figoutput/miss_rate_hist.png", dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.show()

#%%



#%%
print(
    df_llama3["miss_rate"].mean(),
    df_llama3["miss_rate"].std(),
    df_qwen["miss_rate"].mean(),
    df_qwen["miss_rate"].std(),
)

#%%
# hist of miss rate group by num_active_tokens
plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
count_act_tokens = df_llama3["num_active_tokens"].unique()
print(count_act_tokens)
for _i, act in enumerate(count_act_tokens):
    plt.subplot(1, 4, _i+1)
    plt.hist(df_llama3[df_llama3["num_active_tokens"] == act]["miss_rate"], bins=100)
    plt.xlabel(f"num_active_tokens={act}")
# plt.xlabel("Llama-3-8B-1M")
plt.show()


#%%
# find out miss_rate < 0.1 
# df_llama3_1 = df_llama3[df_llama3["miss_rate"] < 0.1]
# df_qwen_1 = df_qwen[df_qwen["miss_rate"] < 0.1]

# print(df_llama3_1.hist())
# # print(df_qwen_1)


#%%
# # concat 
# cat_oKs = [None] * len(oK_singular_values) # type: list[np.ndarray]
# # cat_oVs = [None] * len(oV_singular_values) # type: list[np.ndarray]

# for layer, caches in oK_singular_values.items():
#     cat_oKs[layer] = np.concatenate([
#         c for c in caches if c.shape[-1] == 128
#     ], axis=0)

# # shape (num_samples, num_layers, num_head, num_singular_values)
# oks_cat = np.stack(cat_oKs, axis=1)
# # (num_layers, num_singular_values)
# oks_cat_m = np.mean(oks_cat, axis=(0,2))


# #%%

# colors = ['#77CFF2', '#0D8BD9']
# cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

# fig, ax = plt.subplots(figsize=(8, 4))
# # Get colors for each line
# line_colors = cmap(np.linspace(0, 1, oks_cat_m.shape[0]))

# # Plot each line with its corresponding color
# for i, y in enumerate(oks_cat_m):
#     plt.plot(y, color=line_colors[i])

# sm = ScalarMappable(cmap=cmap)
# sm.set_array(np.arange(oks_cat_m.shape[0]))  # Needed for colorbar scaling
# cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
# cbar.set_label("Layer Index")
# plt.savefig(f"./figoutput/{model_abbr}_oks_mean.png",
#             dpi=300, bbox_inches="tight", pad_inches=0.0)
# plt.show()

#%%
