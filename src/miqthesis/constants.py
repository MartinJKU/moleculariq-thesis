SYSTEM_PROMPT = (
    "You are an expert chemist. Provide answers in <answer>JSON</answer> format."
)

# Qwen chat turn-end token. The model is trained to close every assistant turn with
# <|im_end|>, so it must be an EOS token at evaluation. A raw training checkpoint
# inherits the base model's generation_config, whose eos is <|endoftext|> (151643)
# only; under vLLM the turn never stops, the model over-generates a whole fake
# conversation, and answer extraction collapses. Evaluations must run on checkpoints
# whose generation_config lists this token (prepare_checkpoint guarantees it).
IM_END_TOKEN_ID = 151645

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
    "eos_token_id": [IM_END_TOKEN_ID, 151643],
    "pad_token_id": 151643,
    "use_cache": True,
}

TASK_FAMILIES = ("count", "index", "generation")
