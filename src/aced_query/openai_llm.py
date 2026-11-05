"""OpenAI-compatible chat backend (OpenAI, vLLM, and other compatible servers)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Per-model sampling presets. Matched by case-insensitive substring against
# model_name; first match wins, so order from most-specific to most-general.
_MODEL_PRESETS: List[Dict[str, Any]] = [
    {
        # Qwen3.6 thinking-mode recommended sampling (qwenlm.github.io)
        "match": "qwen3.6",
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"top_k": 20, "min_p": 0.0},
    },
    {
        # gpt-oss recommended sampling
        "match": "gpt-oss",
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {"top_k": 0, "min_p": 0.0},
    },
    {
        # DeepSeek V4 Pro thinking mode — enables internal reasoning.
        # reasoning_effort + thinking.type=enabled are passed via extra_body so
        # they survive any OpenAI-SDK kwarg validation.
        "match": "deepseek-v4-pro",
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {
            "reasoning_effort": "high",
            "thinking": {"type": "enabled"},
        },
    },
]


def resolve_preset(model_name: str) -> Optional[Dict[str, Any]]:
    """Return the first preset whose 'match' is a case-insensitive substring of model_name."""
    name = (model_name or "").lower()
    for preset in _MODEL_PRESETS:
        if preset["match"].lower() in name:
            return preset
    return None


@dataclass
class OpenAILLM:
    """
    LLM wrapper that calls an OpenAI-compatible API.

    Same ``generate(system_prompt, user_prompt)`` interface as the other backends.

    Args:
        model_name: Model identifier sent to the API (e.g. "gpt-oss-20b").
        base_url: API base URL.  Falls back to env var ``OPENAI_BASE_URL``.
        api_key: API key.  Falls back to env var ``OPENAI_API_KEY``.
        max_new_tokens: Default max tokens to generate.
        temperature: Sampling temperature. None → use model preset (else 0.3).
        top_p: Nucleus sampling parameter. None → use model preset (else 0.9).
        extra_body: Extra params forwarded to the API (top_k, min_p, etc.).
            None → use model preset's extra_body.
        use_preset: If True (default), auto-apply per-model sampling presets
            for known models (Qwen3.6, gpt-oss). Explicit kwargs win.
        capture_reasoning: If True, store response.choices[0].message.reasoning_content
            on self.last_reasoning so callers can log the chain of thought.
    """
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_new_tokens: int = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    extra_body: Optional[Dict[str, Any]] = None
    use_preset: bool = True
    capture_reasoning: bool = True

    client: Any = field(default=None, init=False, repr=False)
    last_reasoning: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from openai import OpenAI

        resolved_base = self.base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        # Resolve sampling params: explicit kwargs win, else preset, else defaults.
        preset = resolve_preset(self.model_name) if self.use_preset else None
        if self.temperature is None:
            self.temperature = preset["temperature"] if preset else 0.3
        if self.top_p is None:
            self.top_p = preset["top_p"] if preset else 0.9
        if self.extra_body is None and preset and "extra_body" in preset:
            self.extra_body = dict(preset["extra_body"])

        self.client = OpenAI(
            base_url=resolved_base,
            api_key=resolved_key,
        )
        logger.info(
            f"OpenAILLM ready — model={self.model_name}, base_url={resolved_base}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"extra_body={self.extra_body}, preset={'yes' if preset else 'no'}"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response via the OpenAI chat-completions API."""
        return self.generate_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_new_tokens=max_new_tokens,
        )

    def generate_messages(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from a full message list (supports multi-turn)."""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if self.capture_reasoning:
            self.last_reasoning = reasoning
        content = msg.content or ""
        if not content.strip() and reasoning:
            logger.warning(
                "OpenAILLM: content is empty but reasoning_content is present "
                "(%d chars) — using reasoning_content as response.",
                len(reasoning),
            )
            content = reasoning
        return content.strip()
