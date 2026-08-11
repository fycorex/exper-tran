from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatMessages:
    prompt_only: tuple[dict[str, object], ...]
    with_answer: tuple[dict[str, object], ...]


def classification_messages(prompt: str, answer: str = "7") -> ChatMessages:
    user = {
        "role": "user",
        "content": (
            {"type": "image"},
            {"type": "text", "text": prompt},
        ),
    }
    assistant = {"role": "assistant", "content": answer}
    return ChatMessages((user,), (user, assistant))


def render_chat_template(
    processor: object,
    messages: tuple[dict[str, object], ...],
    *,
    add_generation_prompt: bool,
) -> str:
    """Render the same non-thinking assistant prefix for proxy and target paths."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return processor.apply_chat_template(
            list(messages), **kwargs, enable_thinking=False
        )
    except TypeError:
        return processor.apply_chat_template(list(messages), **kwargs)
