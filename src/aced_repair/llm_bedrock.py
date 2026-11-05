"""AWS Bedrock (Converse API) backend."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

DEGENERATE_STOPS = ("</invoke>\n\n<parameter", "</invoke>\n<parameter")


@dataclass
class BedrockLLM:
    model_id: str
    region_name: Optional[str] = None
    max_new_tokens: int = 1536
    temperature: float = 0.3
    top_p: float = 0.9

    client: Any = field(default=None, init=False, repr=False)
    _tls: Any = field(default_factory=threading.local, init=False, repr=False)

    @property
    def last_finish_reason(self) -> Optional[str]:
        return getattr(self._tls, "finish_reason", None)

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return getattr(self._tls, "usage", None)

    def __post_init__(self):
        region = self.region_name or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        if os.environ.get("BEDROCK_ALLOW_EC2_METADATA", "").lower() not in {"1", "true", "yes"}:
            os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        if not (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or boto3.Session().get_credentials()):
            raise SystemExit(f"{self.model_id} runs on AWS Bedrock, but no AWS credentials were found. Set "
                             "AWS_BEARER_TOKEN_BEDROCK and AWS_DEFAULT_REGION, or configure the AWS credential chain "
                             "(for an EC2/ECS instance role, also set BEDROCK_ALLOW_EC2_METADATA=1).")
        connect_timeout = float(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "10"))
        read_timeout = float(os.environ.get("BEDROCK_READ_TIMEOUT", "900"))   # long replies are not streamed
        max_attempts = int(os.environ.get("BEDROCK_MAX_ATTEMPTS", "3"))
        config = Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"max_attempts": max_attempts, "mode": "standard"},
        )
        self.client = boto3.client(
            "bedrock-runtime", region_name=region, config=config,
        )
        session = boto3.Session()
        credentials = session.get_credentials()
        credential_method = getattr(credentials, "method", None) if credentials is not None else None
        logger.info(
            f"BedrockLLM ready — model={self.model_id}, region={region}, "
            f"read_timeout={read_timeout}s, credential_method={credential_method}, "
            f"AWS_PROFILE_set={bool(os.environ.get('AWS_PROFILE'))}, "
            f"AWS_BEARER_TOKEN_BEDROCK_set={bool(os.environ.get('AWS_BEARER_TOKEN_BEDROCK'))}, "
            f"AWS_EC2_METADATA_DISABLED={os.environ.get('AWS_EC2_METADATA_DISABLED')}"
        )

    def _to_bedrock_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple:
        system_blocks: List[Dict[str, Any]] = []
        converse_msgs: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_blocks.append({"text": text})
            else:
                content = [{"text": text}]
                # ablation (full history): a cache point after the conversation prefix, Claude models only
                if msg.get("cache") and "anthropic.claude" in self.model_id.lower():
                    content.append({"cachePoint": {"type": "default"}})
                converse_msgs.append({
                    "role": role,
                    "content": content,
                })

        return system_blocks, converse_msgs

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
        system_blocks, converse_msgs = self._to_bedrock_messages(messages)
        inference_config: Dict[str, Any] = {
            "maxTokens": max_new_tokens or self.max_new_tokens,
        }
        # Claude models can fall into an endless "</invoke>\n<parameter ...>x</parameter>" loop after a first
        # tool-call block (until max tokens). Stop there; a normal reply never contains these strings.
        if "anthropic.claude" in self.model_id.lower():
            inference_config["stopSequences"] = list(DEGENERATE_STOPS)
        if self._supports_temperature() and self.temperature is not None:
            inference_config["temperature"] = self.temperature

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": converse_msgs,
            "inferenceConfig": inference_config,
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        response = self.client.converse(**kwargs)
        stop = response.get("stopReason")
        self._tls.finish_reason = {"max_tokens": "length", "content_filtered": "content_filtered"}.get(stop, "stop")
        self._tls.usage = response.get("usage")
        blocks = response["output"]["message"].get("content", []) or []
        # keep the model's own reasoning text when the API returns it (not redacted)
        rtxt = [((b.get("reasoningContent") or {}).get("reasoningText") or {}).get("text") for b in blocks]
        self._tls.reasoning = "\n".join(t for t in rtxt if t) or None
        # reasoning models on Bedrock (GPT-6, Claude Haiku 5.5) can return interleaved reasoning and text blocks:
        # a draft answer, more reasoning, then the final answer. Use the last text block.
        texts = [b["text"] for b in blocks if "text" in b and b["text"].strip()]
        self._tls.text_blocks = texts if len(texts) > 1 else None   # kept in the turn record for audit
        text = (texts[-1] if texts else "").strip()
        if not text:
            raise RuntimeError(f"no text in Bedrock reply (stopReason={stop})")
        return text

    @property
    def last_reasoning(self) -> Optional[str]:
        return getattr(self._tls, "reasoning", None)

    @property
    def last_text_blocks(self) -> Optional[List[str]]:
        """All text blocks of the last reply when there were several (None otherwise)."""
        return getattr(self._tls, "text_blocks", None)

    def _supports_temperature(self) -> bool:
        model_id = self.model_id.lower()
        blocked = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-4-9", "claude-opus-5", "claude-sonnet-5",
                   "openai.gpt-6")
        return not any(b in model_id for b in blocked)
