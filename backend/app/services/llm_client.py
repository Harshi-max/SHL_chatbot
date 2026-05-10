import os
from typing import Any

import requests


class LLMClient:
    def __init__(self) -> None:
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

    def generate(self, prompt: str, timeout: int = 12) -> str | None:
        # Prefer Gemini if available.
        if self.gemini_key:
            reply = self._call_gemini(prompt, timeout=timeout)
            if reply:
                return reply
        if self.openrouter_key:
            return self._call_openrouter(prompt, timeout=timeout)
        return None

    def _call_gemini(self, prompt: str, timeout: int) -> str | None:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent"
        )
        resp = requests.post(
            f"{url}?key={self.gemini_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=timeout,
        )
        if not resp.ok:
            return None
        data: dict[str, Any] = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        return parts[0].get("text")

    def _call_openrouter(self, prompt: str, timeout: int) -> str | None:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=timeout,
        )
        if not resp.ok:
            return None
        data: dict[str, Any] = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")
