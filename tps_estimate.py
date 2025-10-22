# %%
import json
import os

import torch
import transformers
from opencompass.models import HuggingFacewithChatTemplate
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from transformers.models.qwen2 import modeling_qwen2

import lrqk.wrap.chat_lrqk as chat_lrqk
from lrqk_attention import *

import multiprocessing as mp
import argparse
from tqdm.auto import tqdm

data_paths = {
    "4k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_8k.json",
    "8k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_8k.json",
    "16k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_16k.json",
    "32k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_32k.json",
    "64k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_64k.json",
    # "128k": "./outputs/empty/predictions/empty/ruler_qa_hotpotqa_128k.json",
}

LLM_MODEL_NAME_PATH = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"

def iter_json(src: str):
    with open(src, "r") as f:
        data = json.load(f)
        for key in tqdm(data):
            yield data[key]['origin_prompt']

def save_buff(name: str, buff: dict):
    with open(f"./outputs/tps_{name}.json", "w") as f:
        json.dump(buff, f, indent=4, ensure_ascii=False)

def tps_estimate(name: str, model: HuggingFacewithChatTemplate, past_key_values_factory=None, *, _dkey: str = None):
    wrapper, _ = WrapForwardTimer.wrap(model.model)
    result_dict = {}

    for dkey, kpath in data_paths.items():
        if _dkey is not None and _dkey != dkey:
            continue
        out_path = f"./outputs/tps_{name}_{dkey}.json"
        if os.path.exists(out_path):
            continue
        print(dkey)
        prefill_tps = []
        decode_tps = []
        try:
            for prompt in iter_json(kpath):
                wrapper.reset()
                with torch.no_grad():
                    if past_key_values_factory is not None:
                        model.generate(
                            [prompt],
                            max_out_len=256,
                            past_key_values=past_key_values_factory(),
                        )
                    else:
                        model.generate(
                            [prompt],
                            max_out_len=256,
                        )

            prefill_tps.append(wrapper.prefill_tps())
            decode_tps.append(wrapper.decode_tps())

        except Exception as e:
            print(e)
            save_buff(f"{name}_{dkey}", {"error": str(e)})
            continue

        result_dict[dkey] = {
            "prefill_tps": np.mean(prefill_tps),
            "decode_tps": np.mean(decode_tps),
        }
        save_buff(f"{name}_{dkey}", result_dict[dkey])
    wrapper.remove_wrapper()
    return result_dict

def tps_estimate_error(model: HuggingFacewithChatTemplate, past_key_values_factory=None):
    try:
        return tps_estimate(model, past_key_values_factory)
    except Exception as e:
        return {"error": str(e)}

def lrqk_factory(
    factory: LightAttentionIndicesFactory,
):
    model = chat_lrqk.LRQKChatBot(
        # path="Qwen/Qwen2.5-7B-Instruct",
        path=LLM_MODEL_NAME_PATH,
        # path="./llama_hf_mirror/llama_hf/Meta-Llama-3.1-8B-Instruct",
        lrqk_num_active_tokens=1024,
        lwattn_factory=factory,
        lrqk_rank=16,
        lrqk_lite_tokens=16,
    )

    return model

def gpu_only_factory():
    model = HuggingFacewithChatTemplate(
        abbr="llama-3-8B-1M",
        path=LLM_MODEL_NAME_PATH,
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    )
    return model

task_dict = {
    "lrqk_default": LightAttentionIndicesFactory,
    "lrqk_offload": LightAttentionIndicesOffloadPrefill,
    "lrqk_no_hitmiss": LightAttentionIndicesNoHitMiss,
    "gpu_only": None,
    "cpu_offload": None,
}

def run_task(name: str, _dkey: str = None):
    out_name = f"./outputs/tps_estimate_{name}.json"
    if os.path.exists(out_name):
        print(f"{out_name} already exists, skipping")
        return
    if name == "cpu_offload":
        records = tps_estimate(name, gpu_only_factory(), FullOffloadCache, _dkey=_dkey)
    elif name == "gpu_only":
        records = tps_estimate(name, gpu_only_factory(), _dkey=_dkey)
    elif name in task_dict:
        task = task_dict[name]
        records = tps_estimate(name, lrqk_factory(task), _dkey=_dkey)
    else:
        raise ValueError(f"Unknown task {name}")
    
    with open(out_name, "w") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    _args = argparse.ArgumentParser()
    _args.add_argument("--task", type=str, default=None)
    _args.add_argument("--dkey", type=str, default=None)
    args = _args.parse_args()
    if args.task is None:
        for task in task_dict.keys():
            run_task(task)
    else:
        run_task(args.task, _dkey=args.dkey)

