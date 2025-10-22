
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
import kvquant_lib as kvquant


@MODELS.register_module()
class KVQuantChatBot(HuggingFacewithChatTemplate):

    def __init__(
        self,
        path: str,
        kvquant_config: Dict,
        **kwargs,
    ):
        if kvquant_config is not None:
            if kvquant_config.get('model_name_or_path', None) != path:
                warnings.warn(f"kvquant_config.model_name_or_path ({kvquant_config.get('model_name_or_path', None)}) is not the same as path ({path}).")
                kvquant_config['model_name_or_path'] = path

        self.kvquant_config = kvquant.KVQuantConfig.from_dict(kvquant_config)
        
        super().__init__(path, **kwargs)

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):

        assert self.kvquant_config.model_name_or_path == path, "kvquant_config.model_name_or_path must be the same as path"

        print(f"Loading KVQuant model from {path} with config {self.kvquant_config}")
        self.model = kvquant.create_kvquant_model(self.kvquant_config, device='cuda')
        self.model.eval()
        self.model.generation_config.do_sample = False
