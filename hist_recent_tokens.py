# %%
from tools.wikitext import WikiTextWrapper
import torch
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation.utils import GenerationConfig
from transformers import PreTrainedModel
import copy
import numpy as np

from collections import defaultdict
import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%


@dataclass
class GenerationParams:
    conversation: list[dict[str, str]] = field(default_factory=list)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    add_generation_prompt: bool = True
    output_attention: bool = True


class TextGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager"
        ).to(self.device).eval()  # type: PreTrainedModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left")

        self.generation_config = GenerationConfig.from_pretrained(
            model_name,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=2048,
        )

        print(type(self.generation_config))

        self.model._prepare_special_tokens(
            self.generation_config,
            kwargs_has_attention_mask=True,
            device=self.device,
        )

    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers

    def _prepare_inputs(self, params: GenerationParams) -> dict:
        return self.tokenizer.apply_chat_template(
            params.conversation,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=params.add_generation_prompt,
            return_dict=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

    def generate_stream(
        self,
        params: GenerationParams
    ):
        # Prepare initial inputs
        inputs = self._prepare_inputs(params)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Configure generation parameters
        gen_config = copy.deepcopy(
            self.generation_config)
        gen_config.max_new_tokens = params.max_new_tokens
        gen_config.temperature = params.temperature
        gen_config.top_p = params.top_p
        gen_config.top_k = params.top_k
        gen_config.do_sample = params.do_sample

        # Initialize generation state
        logits_processor = self.model._get_logits_processor(
            generation_config=gen_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=[],
            device=self.device,
        )

        forward_kwargs = {
            "output_attentions": True,
            "attention_mask": attention_mask,
            "past_key_values": DynamicCache(),
            "cache_position": torch.arange(input_ids.shape[1], device=self.device),
        }

        is_prefill = True

        # Generation loop
        while input_ids.shape[1] < gen_config.max_length:
            # Prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids,
                **forward_kwargs,
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**model_inputs, return_dict=True)

            forward_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs=forward_kwargs, is_encoder_decoder=False)

            # Process logits
            next_logits = logits_processor(
                input_ids, outputs.logits[:, -1, :].clone().float())
            next_token = self._sample_token(next_logits, params.do_sample)

            # Update input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Prepare output
            if not is_prefill:
                yield outputs.attentions
            is_prefill = False

            # Check stopping condition
            if next_token.item() == self.tokenizer.eos_token_id:
                break

    def _sample_token(self, logits: torch.Tensor, do_sample: bool) -> torch.Tensor:
        if do_sample:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

# %%


# %%

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

model_abbr = _model_meta[0]["model_abbr"]
generator = TextGenerator(
    model_name=_model_meta[0]["model_name"],
)

# %%
from tqdm.auto import tqdm

hist_results = [
    []
    for _ in range(generator.num_layers)
]

max_neighbour = 128
dataset = WikiTextWrapper()

for data in tqdm(dataset):
    params = GenerationParams(
        conversation = [
            {"role": "system", "content": "Summarize the following text."},
            {"role": "user", "content": data["content"]},
        ],
        max_new_tokens=128,
        output_attention=True,
    )

    for attns in generator.generate_stream(params):
        # type tuple of (bsz, num_heads, q_len, seq_len)
        # bsz == 0
        _, num_heads, q_len, seq_len = attns[0].shape

        if seq_len < max_neighbour:
            continue

        for li, attn in enumerate(attns):
            # (num_heads, q_len, seq_len)
            attn_q = attn[0, :, -1, -max_neighbour:].cpu().float().numpy()
            # reverse attn_q in the last axis
            # attn_q = np.flip(attn_q, axis=-1)
            # cumsum_attn_q = np.cumsum(attn_q, axis=-1)
            hist_results[li].append(attn_q)

# %%

hist_results_np = [
    np.stack(hist_results[li], axis=0).mean(axis=0)
    for li in range(len(hist_results))
]

# %%
hist_results_np[0].shape

# %%
import matplotlib.pyplot as plt


colors = ['#77CFF2', '#0D8BD9']
cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

fig, ax = plt.subplots(figsize=(8, 4))
# Get colors for each line
line_colors = cmap(np.linspace(0, 1, len(hist_results_np)))

# Plot each line with its corresponding color
for i, y in enumerate(hist_results_np):
    plt.plot(np.mean(y[:, -16:], axis=0), color=line_colors[i])

sm = ScalarMappable(cmap=cmap)
sm.set_array(np.arange(len(hist_results_np)))  # Needed for colorbar scaling
cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label("Layer Index")
plt.xticks(
    [0, 8, 15],
    ["-16", "-8", "current"]
)
plt.savefig(f"./figoutput/{model_abbr}_hit_recent.png",
            dpi=300, bbox_inches="tight", pad_inches=0.0)
plt.show()

# %%
