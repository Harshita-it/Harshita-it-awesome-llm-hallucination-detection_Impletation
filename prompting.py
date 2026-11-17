"""Prompt construction (protocol Section 2)."""

PROMPT_TEMPLATE = (
    "Answer the following question with a short answer only (a few words, no explanation).\n"
    "Question: {question}\nAnswer:"
)


def render_chat_prompt(tokenizer, question: str) -> str:
    """Render the model's chat template as text (version-robust; no tensors returned here).

    NOTE: some chat templates insert a default system message when none is given.
    Print the rendered text once and record it in the log (it is part of the frozen prompt).
    """
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question.strip())}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_input_ids(tokenizer, question: str):
    """1-D LongTensor of prompt token ids. add_special_tokens=False because the template already adds them."""
    text = render_chat_prompt(tokenizer, question)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
