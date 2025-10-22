"""
compute average time requirements for 
- GPU only; full
- CPU only; full
- GPU only; LRQK (no offload)
- CPU only; LRQK (all offload)
- LRQK (standard method)
"""

#%%
import itertools
import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, cache_utils)

import lrqk_attention
from tools.wikitext import WikiTextWrapper

#%%


_model_meta = [
    {
        "model_name": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "model_abbr": "llama-3-8B-1M",
    },
    {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_abbr": "qwen-2.5-7B",
    }
]


def run_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cache_factory: callable,
):
    dataset = WikiTextWrapper()

    time_list = []

    tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
            past_key_values = cache_factory()
            _start = time.time()
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=64,
                do_sample=True,
            )
            _end = time.time()

            _new_generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            time_list.append((_end - _start) / _new_generated_tokens)

    return time_list

def on_gpu_only(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()


    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    def cache_factory():
        return cache_utils.DynamicCache()
    
    return run_loop(model, tokenizer, cache_factory)
    

def on_cpu_only(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left")
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    

    def cache_factory():
        return cache_utils.OffloadedCache()

    return run_loop(model, tokenizer, cache_factory)


#%%

def on_lrqk_standard(model_name: str):

    model, tokenizer = lrqk_attention.load_lrqk_model(model_name, device="cuda:0")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    def cache_factory():
        return lrqk_attention.DynamicLRQKCache(
            r=8,
            num_active_tokens=1024,
            lite_tokens=32,
            tol=0.01,
            max_iter=2,
            num_key_value_groups=num_key_value_groups,
        )

    return run_loop(model, tokenizer, cache_factory=cache_factory)

#%%

# on_gpu_only(_model_meta[0]["model_name"])
# on_cpu_only(_model_meta[0]["model_name"])
on_lrqk_standard(_model_meta[0]["model_name"])

#%%




    







    




