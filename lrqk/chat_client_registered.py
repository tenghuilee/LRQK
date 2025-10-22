
import lrqk.chat_client as chat_client
import lrqk.chat_templates as chat_templates
from functools import partial

# llm_name_or_path = ""

Qwen2_7B = partial(
    chat_client.GenerateClient,
    base_model_or_name="Qwen/Qwen2.5-7B-Instruct",
    chat_template=None,
    chat_template_continue=chat_templates.qwen2_continue_chat_template,
)

# following https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
# the original Qwen2.5-7B-Instruct is limited to 32768 tokens,
# extended to 128k tokens via YaRN
Qwen2_7B_128k = partial(
    chat_client.GenerateClient,
    base_model_or_name="Qwen/Qwen2.5-7B-Instruct",
    chat_template=None,
    chat_template_continue=chat_templates.qwen2_continue_chat_template,
    ext_auto_model_init_kwargs={
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }
    },
)

Llama3_8B = partial(
    chat_client.GenerateClient,
    base_model_or_name="./llama_hf_mirror/llama_hf/llama-3-8b-instruct",
    chat_template=None,
    chat_template_continue=chat_templates.llama3_continue_chat_template,
    eos_token="<|eot_id|>",
    pad_token="<|eot_id|>",
)

Llama2_7B = partial(
    chat_client.GenerateClient,
    base_model_or_name="./llama_hf_mirror/llama_hf/llama-2-7b-chat",
    chat_template=chat_templates.llama2_chat_template,
    chat_template_continue=chat_templates.llama2_continue_chat_template,
)

registed_dict = {
    "Qwen2_7B": Qwen2_7B,
    "Qwen2_7B_128k": Qwen2_7B_128k,
    "Llama3_8B": Llama3_8B,
    "Llama2_7B": Llama2_7B,
}

