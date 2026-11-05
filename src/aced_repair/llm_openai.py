#!/usr/bin/env python3
"""OpenAI-compatible backend (OpenAI, vLLM, other compatible servers)."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("llm_openai")

_MODEL_PRESETS: List[Dict[str, Any]] = [
    {"matches": ["qwen3.6", "qwen3-small", "qwen3-8b", "qwen3"], "temperature": 1.0, "top_p": 0.95,
     "extra_body": {"top_k": 20, "min_p": 0.0}},
    {"matches": ["gpt-oss"], "temperature": 1.0, "top_p": 1.0,
     "extra_body": {"top_k": 0, "min_p": 0.0}},
    # providers' recommended sampling for Gemini 3.x and Gemma 4
    {"matches": ["gemini"], "temperature": 1.0, "top_p": 0.95},
    {"matches": ["gemma"], "temperature": 1.0, "top_p": 0.95, "extra_body": {"top_k": 64}},
    {"matches": ["deepseek-v4-pro"], "temperature": 1.0, "top_p": 0.95,
     "extra_body": {"reasoning_effort": "high", "thinking": {"type": "enabled"}}},
]


def resolve_preset(model_name: str) -> Optional[Dict[str, Any]]:
    name = (model_name or "").lower()
    for preset in _MODEL_PRESETS:
        if any(m.lower() in name for m in preset["matches"]):
            return preset
    return None


@dataclass
class OpenAILLM:
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
    _tls: Any = field(default_factory=threading.local, init=False, repr=False)

    @property
    def last_reasoning(self) -> Optional[str]:
        return getattr(self._tls, "reasoning", None)

    @property
    def last_finish_reason(self) -> Optional[str]:
        return getattr(self._tls, "finish_reason", None)

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return getattr(self._tls, "usage", None)

    @property
    def last_served_model(self) -> Optional[str]:
        return getattr(self._tls, "served_model", None)

    def __post_init__(self):
        from openai import OpenAI

        resolved_base = self.base_url or os.environ.get("OPENAI_BASE_URL")
        resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        preset = resolve_preset(self.model_name) if self.use_preset else None
        if self.temperature is None:
            self.temperature = preset["temperature"] if preset else 0.3
        if self.top_p is None:
            self.top_p = preset["top_p"] if preset else 0.9
        if self.extra_body is None and preset and "extra_body" in preset:
            self.extra_body = dict(preset["extra_body"])

        self.client = OpenAI(base_url=resolved_base, api_key=resolved_key)
        logger.info(
            "OpenAILLM ready — model=%s base_url=%s temp=%s top_p=%s preset=%s",
            self.model_name, resolved_base, self.temperature, self.top_p,
            "yes" if preset else "no",
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: Optional[int] = None) -> str:
        return self.generate_messages(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_new_tokens=max_new_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, Any]],
                          max_new_tokens: Optional[int] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        # retry an empty reply (e.g. Gemini's MALFORMED_FUNCTION_CALL) up to twice before accepting it
        for attempt in range(3):
            response = self.client.chat.completions.create(**kwargs)
            fr = response.choices[0].finish_reason
            msg = response.choices[0].message
            if (msg.content or "").strip() or attempt == 2:
                break
            logger.warning("empty reply (finish_reason=%s); retrying (%d/2)", fr, attempt + 1)
        self._tls.finish_reason = fr
        self._tls.served_model = getattr(response, "model", None)
        try:
            self._tls.usage = response.usage.model_dump() if response.usage else None
        except Exception:
            self._tls.usage = None
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if self.capture_reasoning:
            self._tls.reasoning = reasoning
        content = msg.content or ""
        if not content.strip() and reasoning:
            logger.warning("empty content, using reasoning_content (%d chars)", len(reasoning))
            content = reasoning
        return content.strip()
