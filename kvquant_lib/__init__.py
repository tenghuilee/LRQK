
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.models.llama import modeling_llama

from .kvquant.datautils import get_loaders
from .kvquant.modelutils import find_layers
from .kvquant.simquant_module_quantizer import *


@dataclass
class KVQuantConfig:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models; Only Llama Tested",
            "description": "Specify the path to the pretrained model or its identifier from Hugging Face models. This parameter is required for quantization and should point to a Llama model."
        },
    )

    seed: int = field(
        default=0,
        metadata={
            "help": "random seed",
            "description": "Sets the random seed for reproducibility. Use this to ensure consistent results across runs."
        },
    )

    nsamples: int = field(
        default=16,
        metadata={
            "help": "Number of samples to use for calibration",
            "description": "Determines how many samples from the dataset will be used to calibrate the quantization process. More samples can lead to more accurate quantization."
        },
    )

    abits: int = field(
        default=4,
        metadata={
            "help": "Number of bits for activation quantization",
            "description": "Specifies the number of bits to use for quantizing activations. Lower values reduce precision but save memory."
        },
    )

    nuq: bool = field(
        default=True,
        metadata={
            "help": "Enable non-uniform quantization",
            "description": "Toggle for using non-uniform quantization, which can improve accuracy compared to uniform quantization."
        },
    )

    nf: bool = field(
        default=False,
        metadata={
            "help": "Enable normalization",
            "description": "Toggle to apply normalization during quantization. This can help in maintaining the distribution of values."
        },
    )

    perchannel: List[str] = field(
        default_factory=lambda: ["k_proj"],
        metadata={
            "help": "Layers to quantize per-channel",
            "description": "List of layer names that should be quantized per-channel. Per-channel quantization can lead to better accuracy."
        },
    )

    pertoken: List[str] = field(
        default_factory=lambda: ["v_proj"],
        metadata={
            "help": "Layers to quantize per-token",
            "description": "List of layer names that should be quantized per-token. Per-token quantization can help in maintaining token-specific information."
        },
    )

    include_sparse: bool = field(
        default=True,
        metadata={
            "help": "Include sparse layers in quantization",
            "description": "Toggle to include sparse layers in the quantization process. Sparse layers can significantly reduce memory usage."
        },
    )

    sparsity_threshold: float = field(
        default=0.99,
        metadata={
            "help": "Sparsity threshold for sparse layers",
            "description": "Threshold value to determine sparsity. Values below this threshold will be considered sparse and pruned."
        },
    )

    norm: bool = field(
        default=False,
        metadata={
            "help": "Enable normalization",
            "description": "Toggle to apply normalization during quantization. This can help in maintaining the distribution of values."
        },
    )

    quantizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to save quantizer state",
            "description": "Specifies the file path where the quantizer state will be saved. This is useful for resuming quantization."
        },
    )

    fisher: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to Fisher information matrix",
            "description": "Specifies the path to the Fisher information matrix file. This is used for second-order optimization."
        },
    )

    seqlen: int = field(
        default=2048,
        metadata={
            "help": "Sequence length for calibration",
            "description": "Specifies the sequence length to use during calibration. This should match the expected input sequence length."
        },
    )

    maxseqlen: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length",
            "description": "Specifies the maximum allowed sequence length. This can be used to limit memory usage."
        },
    )

    load: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to load quantized model",
            "description": "Specifies the path to load a previously quantized model. This can be used to skip quantization and directly load a quantized model."
        },
    )

    dataset: str = field(
        default='wikitext2',
        metadata={
            "help": "Dataset to use for calibration",
            "description": "Specifies the dataset to use for calibration. Different datasets may require different calibration settings."
        },
    )

    cap_outliers: float = field(
        default=-1,
        metadata={
            "help": "Cap outliers to this value",
            "description": "Specifies the value to cap outliers to. Set to -1 to disable capping. This can help in reducing the impact of extreme values."
        },
    )

    first_few_fp16: int = field(
        default=-1,
        metadata={
            "help": "Number of layers to keep in FP16",
            "description": "Specifies the number of initial layers to keep in FP16 precision. Set to -1 to disable. This can help in maintaining precision for critical layers."
        },
    )

    clamp: bool = field(
        default=False,
        metadata={
            "help": "Enable clamping",
            "description": "Toggle to enable clamping of values during quantization. Clamping can help in maintaining value ranges."
        },
    )


    def __post_init__(self):

        if self.quantizer_path is not None:
            _folder = os.path.dirname(self.quantizer_path)
            if _folder and not os.path.exists(_folder):
                print(f"Creating folder {os.path.dirname(self.quantizer_path)}")
                os.makedirs(_folder)
    
    @staticmethod
    def from_dict(config_dict: Optional[Dict[str, Any]]):
        if config_dict is None:
            return None
        return KVQuantConfig(**config_dict)

def _monk_patches_torch_init(func):
    def _skip(*args, **kwargs):
        pass

    def __inner(*args, **kwargs):
        omethods = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = _skip
        torch.nn.init.uniform_ = _skip
        torch.nn.init.normal_ = _skip
        
        ans = func(*args, **kwargs)

        # recover
        torch.nn.init.kaiming_uniform_ = omethods[0]
        torch.nn.init.uniform_ = omethods[1]
        torch.nn.init.normal_ = omethods[2]

        return ans
    return __inner


class KVQuantLlama:
    def __init__(self, config: KVQuantConfig):
        self.config = config
        if config.model_name_or_path is None:
            raise ValueError("model_name_or_path is required in config")

    @_monk_patches_torch_init
    def _load_model(self, device=None) -> PreTrainedModel:
        model_config = AutoConfig.from_pretrained(
            self.config.model_name_or_path)
        context_size = self.config.maxseqlen
        orig_ctx_len = getattr(model_config, "max_position_embeddings", None)
        if orig_ctx_len and context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(context_size / orig_ctx_len))
            model_config.rope_scaling = {
                "type": "linear", "factor": scaling_factor}

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            # use_flash_attention_2=True, # old method, removed
            attn_implementation="flash_attention_2",
            torch_dtype=torch.half,
            # torch_dtype=torch.bfloat16,
        ).eval()  # type: PreTrainedModel

        if self.config.seqlen != -1:
            model.seqlen = self.config.seqlen
        if model_config.vocab_size == 32001:  # ?
            model.resize_token_embeddings(32001)

        if device is not None:
            model = model.to(device)

        return model

    def load_fisher(self):
        if self.config.fisher is None:
            return None

        import os

        onlyfiles = [
            os.path.join(self.config.fisher, f)
            for f in os.listdir(self.config.fisher)
            if (('pytorch_model' in f or 'safetensors' in f) and 'index' not in f)
        ]

        mypath = onlyfiles[0]
        if 'safe' in mypath:
            from safetensors.torch import load_file
            fisher = load_file(mypath, device='cpu')
            for i in range(1, len(onlyfiles)):
                d2 = load_file(onlyfiles[i], device='cpu')
                fisher.update(d2)
        else:
            fisher = torch.load(mypath, map_location='cpu')
            for i in range(1, len(onlyfiles)):
                d2 = torch.load(onlyfiles[i], map_location='cpu')
                fisher.update(d2)

        return fisher

    @torch.no_grad()
    def calibration(
        self,
        model: modeling_llama.LlamaForCausalLM,
        dataloader: torch.utils.data.DataLoader,
        fisher=None,
        device=None,
    ):
        # copy from llama_calibration
        print(f'Calibrating {self.config.model_name_or_path}...')
        if device is None:
            device = model.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        layers[0] = layers[0].to(device)

        dtype = next(iter(model.parameters())).dtype
        inps = torch.zeros(
            (self.config.nsamples, model.seqlen,
             model.config.hidden_size), dtype=dtype, device=device
        )
        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
                raise ValueError
        layers[0] = Catcher(layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        print(f'Quantizing {self.config.model_name_or_path}...')

        quantizers = {}
        for i in range(len(layers)):
            print("Layer", i)
            layer = layers[i].to(device)
            full = find_layers(layer)

            perchannel_list = []
            pertensor_list = []
            full_list = []

            for f in full:
                for p in self.config.perchannel:
                    if p in f:
                        perchannel_list.append(f)
                        full_list.append(f)
                for p in self.config.pertoken:
                    if p in f:
                        pertensor_list.append(f)
                        full_list.append(f)

            sequential = list(full.keys())

            simquant = {}  # type: Dict[str, SimQuant]
            subset = {n: full[n] for n in sequential if n in full_list}
            sequential_subset = list(subset.keys())

            for name in sequential:
                if name in perchannel_list:
                    simquant[name] = SimQuant(
                        subset[name],
                        self.config.abits,
                        perchannel=True,
                        qchannel=0,
                    )
                elif name in pertensor_list:
                    simquant[name] = SimQuant(
                        subset[name],
                        self.config.abits,
                        perchannel=True,
                        qchannel=-1,
                    )
                else:
                    continue

            def add_batch(name):
                def tmp(_, inp, out):
                    simquant[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []

            for name in sequential_subset:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name)))

            for j in range(len(dataloader)):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

            for h in handles:
                h.remove()

            for name in subset:

                if fisher is not None:
                    key = f'model.layers.{i}.{name}.weight'
                    fisher_info = fisher[key].cpu()
                else:
                    fisher_info = None

                if "k_proj" in name:
                    if self.config.cap_outliers == -1:
                        cap = False
                    else:
                        cap = True
                else:
                    cap = False

                quantizers[f'model.layers.{i}.{name}'] = simquant[name].quantize(
                    include_sparse=self.config.include_sparse,
                    sparsity_threshold=self.config.sparsity_threshold,
                    nuq=self.config.nuq,
                    fisher=fisher_info,
                    norm=self.config.norm,
                    cap_outliers=cap,
                    first_few_fp16=self.config.first_few_fp16,
                )

                simquant[name].free()

            layers[i] = layer.cpu()
            del layer
            del simquant
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        model.config.use_cache = use_cache
        return quantizers

    @torch.no_grad()
    def run_calibration(self, device=None):
        if self.config.quantizer_path is None:
            raise ValueError("Quantizer path not specified.")

        model = self._load_model(device=device)

        dataloader, testloader = get_loaders(
            self.config.dataset,
            nsamples=self.config.nsamples,
            seed=self.config.seed,
            model=self.config.model_name_or_path,
            seqlen=model.seqlen,
        )

        quantizers = self.calibration(
            model=model,
            dataloader=dataloader,
            fisher=self.load_fisher(),
            device=device,
        )

        with open(self.config.quantizer_path, 'wb') as f:
            pickle.dump(quantizers, f, protocol=pickle.HIGHEST_PROTOCOL)

        del model

    def kvquant_model(self, device=None):
        if not os.path.exists(self.config.quantizer_path):
            print("Quantization file not found. Running calibration ...")
            self.run_calibration(device=device)
            if not os.path.exists(self.config.quantizer_path):
                raise ValueError("Quantization file not created. Please check the calibration process.")

        with open(self.config.quantizer_path, 'rb') as f:
            quantizers = pickle.load(f)

        # replace layers
        perchannelquant = {}
        pertokenquant = {}

        perchannel_match = self.config.perchannel
        pertoken_match = self.config.pertoken

        model = self._load_model(device=device)

        for k in quantizers.keys():
            # quantizers[k] = quantizers[k] + (-1, ) # empty for now (used to be LN params)

            # filter out tensor list
            for p in perchannel_match:
                if p in k:
                    perchannelquant[k] = quantizers[k]

            for p in pertoken_match:
                if p in k:
                    pertokenquant[k] = quantizers[k]

        # per-vector quant
        make_quant_sim(
            model,
            perchannelquant,
            self.config.abits,
            perchannel=True,
            include_sparse=self.config.include_sparse,
            sparsity_threshold=self.config.sparsity_threshold,
            dynamicquantization=False,
            nuq=self.config.nuq,
            nf_nuq=self.config.nf,
            norm=self.config.norm,
            cap_outliers=self.config.cap_outliers,
            first_few_fp16=self.config.first_few_fp16,
            clamp=self.config.clamp
        )

        # per-vector quant
        make_quant_sim(
            model,
            pertokenquant,
            self.config.abits,
            perchannel=False,
            include_sparse=self.config.include_sparse,
            sparsity_threshold=self.config.sparsity_threshold,
            dynamicquantization=True,
            nuq=self.config.nuq,
            nf_nuq=self.config.nf,
            norm=self.config.norm,
            cap_outliers=self.config.cap_outliers,
            first_few_fp16=self.config.first_few_fp16,
            clamp=self.config.clamp
        )

        return model
    
def create_kvquant_model(config, device=None):
    """A quick method to create a quantized model for a given config."""
    return KVQuantLlama(config).kvquant_model(device=device)
    
def create_quantizer(config, device=None):
    """A quick way to create a quantizer for a given model."""
    return KVQuantLlama(config).run_calibration(device=device)
