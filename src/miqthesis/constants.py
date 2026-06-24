SYSTEM_PROMPT = (
    "You are an expert chemist. Provide answers in <answer>JSON</answer> format."
)

QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

CONTROLLED_GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 512,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "use_cache": True,
}

TASK_FAMILIES = ("count", "index", "generation")
