"""
Implementation of a chatbot for OpenCampas
"""

from typing import Dict, List, Optional

import torch
from opencompass.models.huggingface_above_v4_33 import (
    HuggingFacewithChatTemplate, _convert_base_messages,
    _convert_chat_messages, _format_with_fast_chat_template,
    _get_stopping_criteria)
import transformers

import warnings

from opencompass.registry import MODELS
# from lrqk.wrap.registry_lrqk import LRQK_REGISTRY
import lrqk_attention
import kvquant_lib as kvquant

# from opencompass.models.base import BaseModel

@MODELS.register_module()
class LRQKChatBot(HuggingFacewithChatTemplate):

    def __init__(
        self, 
        path: str,
        lrqk_rank: int = 16,
        lrqk_num_active_tokens: int = 1024,
        lrqk_lite_tokens: int = 64,
        lrqk_max_iter: int = 2,
        lrqk_tol: float = 1e-2,
        lrqk_init_aq_ak_method: str = 'randn',
        yarn_config: Optional[Dict] = None,
        lwattn_factory = lrqk_attention.LightAttentionIndicesFactory,
        kvquant_config: Optional[Dict] = None,
        **kwargs,
    ):
        # must init before super().__init__()
        self.yarn_config = yarn_config if yarn_config is not None else dict()

        if kvquant_config is not None:
            if kvquant_config.get('model_name_or_path', None) != path:
                warnings.warn(f"kvquant_config.model_name_or_path ({kvquant_config.get('model_name_or_path', None)}) is not the same as path ({path}).")
                kvquant_config['model_name_or_path'] = path

        self.kvquant_config = kvquant.KVQuantConfig.from_dict(kvquant_config)

        super().__init__(path, **kwargs)
        self.lrqk_rank = lrqk_rank
        self.lrqk_num_active_tokens = lrqk_num_active_tokens
        self.lrqk_max_iter = lrqk_max_iter
        self.lrqk_lite_tokens = lrqk_lite_tokens
        self.lrqk_tol = lrqk_tol
        self.lrqk_init_aq_ak_method = lrqk_init_aq_ak_method

        if lwattn_factory is None:
            lwattn_factory = lrqk_attention.LightAttentionIndicesFactory
        elif isinstance(lwattn_factory, str):
            if lwattn_factory == 'default':
                lwattn_factory = lrqk_attention.LightAttentionIndicesFactory
            elif lwattn_factory == 'offload':
                lwattn_factory = lrqk_attention.LightAttentionIndicesOffloadPrefill
            else:
                raise ValueError(f'Invalid lwattn_factory: {lwattn_factory}')
        self.lwattn_factory = lwattn_factory

        # four lambdas are 1.0
        _conf = self.model.config

        num_key_value_groups = _conf.num_attention_heads // _conf.num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        device = kwargs.get('device', 'cuda:0')
        if device is None:
            device = 'cuda:0'

        if self.kvquant_config is not None:
            # enable kvquant
            assert self.kvquant_config.model_name_or_path == path, f"kvquant_config.model_name_or_path must be the same as path"
            print(f"Loading kvquant model from {self.kvquant_config.model_name_or_path}")
            model = kvquant.create_kvquant_model(self.kvquant_config, device=device)
        else:
            # disable kvquant
            model = None

        self.model = lrqk_attention.load_model_hack(
            path,
            device=device,
            yarn=self.yarn_config,
            base_model=model,
        )

        self.model.eval()

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Adopt from super.generate"""

        messages = _convert_chat_messages(inputs)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.model.device)

        if self.mode == 'mid':
            raise NotImplementedError('mid mode is not supported yet')

        # generation_kwargs = self.generation_kwargs.copy()
        # generation_kwargs.update(kwargs)
        # stopping_criteria = list(set(stopping_criteria + self.stop_words))
        # if stopping_criteria:
        #     generation_kwargs['stopping_criteria'] = _get_stopping_criteria(
        #         stopping_criteria, self.tokenizer, batch_size)
        # if max_out_len is not None:
        #     generation_kwargs['max_new_tokens'] = max_out_len
        # if min_out_len is not None:
        #     generation_kwargs['min_new_tokens'] = min_out_len
        # generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        # self.logger.info('Generation Args of Huggingface: ')
        # self.logger.info(generation_kwargs)

        # step-2: conduct model forward to generate output
        with torch.no_grad():
            in_seq_len = inputs.input_ids.shape[1]
            with torch.autocast("cuda", dtype=self.model.dtype):
                past_key_values = lrqk_attention.DynamicLRQKCache(
                    num_key_value_groups=self.num_key_value_groups,
                    r=self.lrqk_rank,
                    num_active_tokens=self.lrqk_num_active_tokens,
                    lite_tokens=self.lrqk_lite_tokens,
                    tol=self.lrqk_tol,
                    max_iter=self.lrqk_max_iter,
                    lwattn_factory=self.lwattn_factory,
                    init_aq_ak_method=self.lrqk_init_aq_ak_method,
                )
                print(self.lwattn_factory)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_out_len,
                    do_sample=False,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            outputs = outputs.narrow(1, in_seq_len, outputs.shape[1] - in_seq_len)
            del past_key_values

        # step-3: decode the output
        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # for stop in stopping_criteria:
        #     decodeds = [t.split(stop)[0] for t in decodeds]

        return decodeds



