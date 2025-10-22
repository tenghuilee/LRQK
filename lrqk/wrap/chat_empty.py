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

# from opencompass.models.base import BaseModel

@MODELS.register_module()
class EmptyChatBot(HuggingFacewithChatTemplate):

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        pass

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Adopt from super.generate"""
        return ["" for _ in inputs]



