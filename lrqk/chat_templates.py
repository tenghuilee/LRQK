
qwen2_continue_chat_template = """
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- \'<|im_start|>\' + message.role + \'\\n\' + message.content + \'<|im_end|>\' + \'\\n\' }}
    {%- elif message.role == "assistant" %}
        {{- \'<|im_start|>\' + message.role }}
        {%- if message.content %}
            {{- \'\\n\' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- \'\\n<tool_call>\\n{"name": "\' }}
            {{- tool_call.name }}
            {{- \'", "arguments": \' }}
            {{- tool_call.arguments | tojson }}
            {{- \'}\\n</tool_call>\' }}
        {%- endfor %}
        {{- \'<|im_end|>\\n\' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- \'<|im_start|>user\' }}
        {%- endif %}
        {{- \'\\n<tool_response>\\n\' }}
        {{- message.content }}
        {{- \'\\n</tool_response>\' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- \'<|im_end|>\\n\' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- \'<|im_start|>assistant\\n\' }}
{%- endif %}
"""


llama3_continue_chat_template = """
{%- set loop_messages = messages %}
{%- for message in loop_messages %}
    {%- set content = message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}
    {%- if loop.index0 == 0 %}
        {%- set content = bos_token + content %}
    {%- endif %}
    {{- content}}
{%- endfor %}
{%- if add_generation_prompt %}
{{- \'<|start_header_id|>assistant <|end_header_id|>\\n\\n\' }}
{%- endif %}
"""

llama2_chat_template = """
{%- if messages[0]['role'] == 'system' %}
    {%- set loop_messages = messages[1:] %}
    {%- set system_message = messages[0]['content'] %}
{%- elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}
    {%- set loop_messages = messages %}
    {%- set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- set system_message = false %}
{%- endif %}
{%- if loop_messages|length == 0 and system_message %}
    {{- bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}
{%- endif %}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- if loop.index0 == 0 and system_message != false %}
        {%- set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
    {%- else %}
        {%- set content = message['content'] %}
    {%- endif %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + content.strip() + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
"""

llama2_continue_chat_template = """
{%- set loop_messages = messages %}
{%- for message in loop_messages %}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif %}
    {%- set content = message['content'] %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + content.strip() + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
"""

