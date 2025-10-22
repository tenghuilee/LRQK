"""
histogram of the lowrankness of the KV heads in Salesforce/wikitext

"""

#%%

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap


from collections import defaultdict

import torch
import transformers
from torch.utils.data import Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, cache_utils)

# huggingface dataloader
from datasets import load_dataset
import wikitext_q_lowrankness_helper as helper
import argparse

#%%

def convert_to_jsonl(output_file: str):
    # Load the dataset
    wikitext = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-v1",
        split="test",
    )

    # Matches lines like "= Heading ="
    heading1_pattern = re.compile(r'^\s*=([^=]+)=\s*$')


    with open(output_file, "w", encoding="utf-8") as f:
        current_heading = None
        current_lines = []

        for line in wikitext["text"]:
            line = line.strip()
            if not line:
                continue

            heading_match = heading1_pattern.match(line)
            if heading_match:
                # Save previous section if exists
                if current_heading and current_lines:
                    f.write(json.dumps({
                        "heading": current_heading.strip(),
                        "content": "\n".join(current_lines).strip()
                    }) + "\n")

                # Start new section
                current_heading = heading_match.group(1)
                current_lines = []
            else:
                if current_heading:
                    current_lines.append(line)

        # Final section
        if current_heading and current_lines:
            f.write(json.dumps({
                "heading": current_heading.strip(),
                "content": "\n".join(current_lines).strip()
            }) + "\n")

class WikiTextWrapper(Dataset):
    def __init__(self, wikitext_jsonl: str):

        # filter too short sentences
        with open(wikitext_jsonl, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def plot_singular_values(
    svals: dict,
    model_abbr: str,
    tag: str,
    save_dir: str = "./figoutput",
) -> None:
    """
    Plot the mean singular values across layers and heads.
    
    Args:
        svals: Dictionary containing singular values for each layer
        model_abbr: Model abbreviation for the filename
        save_path: Path to save the output figure
    """
    # Concatenate singular values for each layer
    cat_sval = [None] * len(svals)  # type: list[np.ndarray]
    
    for layer, caches in svals.items():
        cat_sval[layer] = np.concatenate([
            c for c in caches if c.shape[-1] == 128
        ], axis=0)

    # Shape (num_samples, num_layers, num_head, num_singular_values)
    # => (num_layers, num_singular_values)
    m_sval = np.stack(cat_sval, axis=1).mean(axis=(0,2)) # type: np.ndarray

    # Create custom colormap
    colors = ['#77CFF2', '#0D8BD9']
    cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    line_colors = cmap(np.linspace(0, 1, m_sval.shape[0]))

    # Plot each line with its corresponding color
    for i, y in enumerate(m_sval):
        plt.plot(y, color=line_colors[i])

    # Add colorbar
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(np.arange(m_sval.shape[0]))
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label("Layer Index")
    
    # Save and show plot
    plt.savefig(
        os.path.join(save_dir, f"{model_abbr}_{tag}_mean.png"),
        dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.show()

def compute_singular_values(dst: dict[int, list], kvcache: list[torch.Tensor]):
    for layer, cache in enumerate(kvcache):
        _, S, _ = torch.svd(cache.float(), compute_uv=False, some=True)
        dst[layer].append(S.cpu().numpy())

def compute_singular_values_cpu(dst: dict[int, list], kvcache: list[torch.Tensor]):
    for layer, cache in enumerate(kvcache):
        cache = cache.float().cpu()
        _, S, _ = torch.svd(cache, compute_uv=False, some=True)
        dst[layer].append(S.numpy())

def parse_args():
    parser = argparse.ArgumentParser(description="Process model and dataset parameters")
    
    # Dataset argument
    parser.add_argument(
        "--wikitext_jsonl",
        type=str,
        default="./datasets/wikitext-2-v1_test.jsonl",
        help="Path to the wikitext jsonl file"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        nargs="*",
        default=None,
        help="List of paths to models or model identifiers from huggingface.co/models, keep none to use default pairs of model_abbr"
    )
    
    parser.add_argument(
        "--model_abbr",
        type=str,
        nargs="*",
        default=["llama-3-8B-1M", "qwen-2.5-7B"],
        help="List of abbreviations for model names (used for saving files)"
    )
    
    # Computation arguments
    parser.add_argument(
        "--compute_svd_in_gpu",
        action="store_true",
        help="Whether to compute SVD on GPU; default is in CPU to save memory"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1,
        help="Maximum number of new tokens to generate"
    )

    model_name_abbr = {
        "llama-3-8B-1M": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "qwen-2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    }

    args = parser.parse_args()

    for abbr in args.model_abbr:
        if abbr not in model_name_abbr:
            raise ValueError(f"Invalid model abbreviation: {abbr}")

    if args.model_name_or_path is None:
        args.model_name_or_path = [model_name_abbr[abbr] for abbr in args.model_abbr]

    return args
    

def main():
    args = parse_args()
    print(args)
    
    # Ensure dataset exists
    if not os.path.exists(args.wikitext_jsonl):
        convert_to_jsonl(args.wikitext_jsonl)

    dataset = WikiTextWrapper(
        args.wikitext_jsonl
    )
    
    for model_name_or_path, model_abbr in zip(args.model_name_or_path, args.model_abbr):
        counting_singular_values(
            model_name_or_path,
            model_abbr,
            args.max_new_tokens,
            args.compute_svd_in_gpu,
            dataset,
        )

def counting_singular_values(
    model_name_or_path: str,
    model_abbr: str,
    max_new_tokens: int,
    compute_svd_in_gpu: bool,
    dataset: WikiTextWrapper,
):
    model = helper.load_model(
        model_name_or_path,
        device="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
    )

    # tokenizer.pad_token = tokenizer.eos_token
    oQ_singular_values = defaultdict(list)
    oK_singular_values = defaultdict(list)
    oV_singular_values = defaultdict(list)

    for src in tqdm(dataset):
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Summarize the following text."},
                {"role": "user", "content": src["content"]},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            # padding=True,
            max_length=4096, # <= standard version
            # max_length=2048,
            return_dict=True,
        )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            past_key_values = helper.WrapDynamicCache()
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

            # compute svd
            if compute_svd_in_gpu:
                # q always in cpu
                compute_singular_values_cpu(oQ_singular_values, past_key_values.query_cache)
                compute_singular_values(oK_singular_values, past_key_values.key_cache)
                compute_singular_values(oV_singular_values, past_key_values.value_cache)
            else:
                compute_singular_values_cpu(oQ_singular_values, past_key_values.query_cache)
                compute_singular_values_cpu(oK_singular_values, past_key_values.key_cache)
                compute_singular_values_cpu(oV_singular_values, past_key_values.value_cache)

        torch.cuda.empty_cache()

    plot_singular_values(
        oK_singular_values,
        model_abbr=model_abbr,
        tag=f"oks_max_token_{max_new_tokens}",
    )

    plot_singular_values(
        oQ_singular_values,
        model_abbr=model_abbr,
        tag=f"oqs_max_token_{max_new_tokens}",
    )
    
if __name__ == "__main__":
    main()
