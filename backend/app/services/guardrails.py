from typing import Iterable


OFF_TOPIC_KEYWORDS = {
    "stock",
    "crypto",
    "medical",
    "therapy",
    "lawsuit",
    "legal",
    "tax",
}
INJECTION_KEYWORDS = {
    "ignore previous",
    "system prompt",
    "developer message",
    "jailbreak",
    "do anything now",
    "pretend you are not restricted",
    "hidden prompt",
}


def is_off_topic(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in OFF_TOPIC_KEYWORDS)


def is_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in INJECTION_KEYWORDS)


def user_turns(messages: Iterable[dict[str, str]]) -> int:
    return sum(1 for msg in messages if msg.get("role") == "user")
