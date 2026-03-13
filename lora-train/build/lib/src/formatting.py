from __future__ import annotations

import json
from typing import Any, Dict, List


VALID_ROLES = {"system", "user", "assistant"}


def _normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    role = str(message.get("role", "")).strip().lower()
    content = str(message.get("content", "")).strip()
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role in chat message: {role!r}")
    if not content:
        raise ValueError("Message content cannot be empty.")
    return {"role": role, "content": content}


def row_to_messages(row: Dict[str, Any], system_prompt: str = "") -> List[Dict[str, str]]:
    """Convert supported row schemas into chat messages."""
    if "messages" in row and row["messages"]:
        raw_messages = row["messages"]
        if isinstance(raw_messages, str):
            raw_messages = json.loads(raw_messages)
        if not isinstance(raw_messages, list):
            raise ValueError("`messages` must be a list.")
        messages = [_normalize_message(m) for m in raw_messages]
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    instruction = str(row.get("instruction", "")).strip()
    model_output = str(row.get("output", row.get("response", ""))).strip()
    if not instruction or not model_output:
        raise ValueError(
            "Instruction format requires non-empty `instruction` and `output` fields."
        )

    input_text = str(row.get("input", "")).strip()
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\nInput:\n{input_text}"

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": model_output})
    return messages


def assistant_response_from_chat(messages: List[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message["content"]
    return ""


def prompt_messages_from_chat(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if messages and messages[-1]["role"] == "assistant":
        return messages[:-1]
    return messages


def _fallback_chat_render(
    messages: List[Dict[str, str]], add_generation_prompt: bool = False
) -> str:
    parts: List[str] = []
    for message in messages:
        parts.append(f"<{message['role']}>\n{message['content']}\n")
    if add_generation_prompt:
        parts.append("<assistant>\n")
    return "\n".join(parts).strip()


def messages_to_text(
    messages: List[Dict[str, str]],
    tokenizer: Any,
    add_generation_prompt: bool = False,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return _fallback_chat_render(
        messages=messages, add_generation_prompt=add_generation_prompt
    )

