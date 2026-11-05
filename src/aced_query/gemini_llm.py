"""
gemini_llm.py

Google Gemini LLM wrapper using the unified google-genai SDK.
Drop-in replacement for OpenAILLM / BedrockLLM — same generate() interface.

Auth (auto-detected by the SDK):
    Google AI Studio (default):
        export GEMINI_API_KEY=...
    Vertex AI:
        export GOOGLE_GENAI_USE_VERTEXAI=true
        export GOOGLE_CLOUD_PROJECT=...
        export GOOGLE_CLOUD_LOCATION=us-central1
        # plus standard ADC (gcloud auth application-default login)

Usage:
    from gemini_llm import GeminiLLM

    llm = GeminiLLM(model_id="gemini-3.1-pro-preview")
    response = llm.generate("You are helpful.", "What is 2+2?")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GeminiLLM:
    """LLM wrapper for Google Gemini via the google-genai SDK.

    Args:
        model_id: Gemini model name (e.g. "gemini-3.1-pro-preview", "gemini-2.5-pro").
        api_key: AI Studio API key. Falls back to env GEMINI_API_KEY / GOOGLE_API_KEY.
            Ignored when GOOGLE_GENAI_USE_VERTEXAI=true (Vertex uses ADC).
        max_new_tokens: Default max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        thinking_budget: Per-call thinking budget in tokens. None = model default
            (dynamic). 0 disables thinking. -1 forces unlimited dynamic thinking.
            Gemini 3 Pro defaults to dynamic thinking; you usually want None here.
        capture_reasoning: If True, store any thought-summary parts on
            self.last_reasoning for downstream logging.
    """

    model_id: str
    api_key: Optional[str] = None
    max_new_tokens: int = 1536
    temperature: float = 1.0
    top_p: float = 0.95
    thinking_budget: Optional[int] = None
    capture_reasoning: bool = True

    client: Any = field(default=None, init=False, repr=False)
    last_reasoning: Optional[str] = field(default=None, init=False, repr=False)
    _types: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        from google import genai
        from google.genai import types as genai_types

        self._types = genai_types

        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
            "true", "1", "yes",
        )
        if use_vertex:
            # SDK reads GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION from env;
            # auth via application-default credentials.
            self.client = genai.Client(vertexai=True)
            backend = "vertex"
        else:
            key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not key:
                raise RuntimeError(
                    "GeminiLLM: no API key found. Set GEMINI_API_KEY (AI Studio) "
                    "or GOOGLE_GENAI_USE_VERTEXAI=true (Vertex)."
                )
            self.client = genai.Client(api_key=key)
            backend = "ai-studio"

        logger.info(
            f"GeminiLLM ready — model={self.model_id}, backend={backend}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"thinking_budget={self.thinking_budget}"
        )

    def _build_config(
        self, system_prompt: Optional[str], max_new_tokens: Optional[int]
    ):
        cfg_kwargs: Dict[str, Any] = {
            "max_output_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if system_prompt:
            cfg_kwargs["system_instruction"] = system_prompt
        if self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=self.capture_reasoning,
            )
        elif self.capture_reasoning:
            cfg_kwargs["thinking_config"] = self._types.ThinkingConfig(
                include_thoughts=True,
            )
        return self._types.GenerateContentConfig(**cfg_kwargs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
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
        """Generate from an OpenAI-style message list. Multi-turn supported.

        Gemini's contents are a list of {role, parts}. Roles: 'user' / 'model'.
        System messages are pulled out into system_instruction.
        """
        system_prompt: Optional[str] = None
        contents: List[Any] = []
        for m in messages:
            role = m["role"]
            text = m["content"]
            if role == "system":
                # Concatenate multiple system messages.
                system_prompt = (system_prompt + "\n\n" + text) if system_prompt else text
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    self._types.Content(
                        role=gemini_role,
                        parts=[self._types.Part(text=text)],
                    )
                )

        config = self._build_config(system_prompt, max_new_tokens)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )

        # Pull out thought summaries (if any) and the final text.
        text_parts: List[str] = []
        thought_parts: List[str] = []
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []) if cand.content else []:
                t = getattr(part, "text", None)
                if not t:
                    continue
                if getattr(part, "thought", False):
                    thought_parts.append(t)
                else:
                    text_parts.append(t)

        if self.capture_reasoning:
            self.last_reasoning = "\n".join(thought_parts) if thought_parts else None

        if text_parts:
            return "".join(text_parts).strip()
        # Fallback to .text convenience accessor (flattens parts but loses thought split).
        return (response.text or "").strip()
