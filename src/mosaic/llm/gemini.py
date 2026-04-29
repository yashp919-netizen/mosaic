"""Gemini 2.0 Flash client via google-genai SDK.

Authentication: GEMINI_API_KEY from environment or .env file.
If the env var is absent, raises RuntimeError with gcloud fallback hint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")
except ImportError:
    pass


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Gemini credentials not found. "
            "Set GEMINI_API_KEY in .env or run: gcloud auth application-default login"
        )
    return key


class _GeminiInstructorShim:
    """Minimal instructor-compatible shim wrapping google-genai SDK."""

    def __init__(self, model: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=_get_api_key())
        self._model_name = model
        self.token_count: int = 0

    # ---- instructor-style interface (duck-typed for MosaicLLMClient) ----

    class _ChatCompletions:
        def __init__(self, outer: "_GeminiInstructorShim") -> None:
            self._outer = outer

        def create(
            self,
            *,
            model: str,
            messages: list[dict],
            response_model: type | None = None,
            **kwargs: Any,
        ) -> Any:
            return self._outer._call(messages, response_model)

    class _Chat:
        def __init__(self, outer: "_GeminiInstructorShim") -> None:
            self.completions = _GeminiInstructorShim._ChatCompletions(outer)

    @property
    def chat(self) -> "_Chat":
        return self._Chat(self)

    # ---- internals ----

    def _call(self, messages: list[dict], response_model: type | None) -> Any:
        from google.genai import types as gtypes

        prompt = self._build_prompt(messages, response_model)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=gtypes.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json" if response_model else "text/plain",
            ),
        )
        text = response.text or ""

        # Track token usage
        try:
            usage = response.usage_metadata
            self.token_count += (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)
        except Exception:
            pass

        if response_model is None:
            return _FakeChoice(text)

        cleaned = self._extract_json(text)
        return response_model.model_validate_json(cleaned)

    @staticmethod
    def _build_prompt(messages: list[dict], response_model: type | None) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"[{role.upper()}]\n{content}")
        prompt = "\n\n".join(parts)
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema(), indent=2)
            prompt += (
                f"\n\n[INSTRUCTION]\nRespond with ONLY valid JSON matching this schema:\n{schema}"
            )
        return prompt

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            text = "\n".join(inner).strip()
        return text


class _FakeChoice:
    """Minimal stand-in for openai ChatCompletion response in raw (non-structured) mode."""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeMessage(content)]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.message = _FakeContent(content)


class _FakeContent:
    def __init__(self, content: str) -> None:
        self.content = content


def get_gemini_shim(model: str = "gemini-2.5-flash") -> _GeminiInstructorShim:
    """Return an instructor-compatible shim for Gemini 2.0 Flash."""
    return _GeminiInstructorShim(model)
