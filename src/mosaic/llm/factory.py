"""LLM client factory.

Returns a MosaicLLMClient for the requested provider — a thin wrapper that
carries the model name so every call site doesn't have to repeat it.

Providers: 'ollama' (local, default), 'gemini' (Vertex AI, Day 9).
"""

from __future__ import annotations

from typing import Any, Literal

import instructor


class MosaicLLMClient:
    """Thin wrapper around an instructor client that pins the model name."""

    def __init__(self, client: instructor.Instructor, model: str) -> None:
        self._client = client
        self._model = model

    def create(self, response_model: type, messages: list[dict], **kwargs: Any) -> Any:
        """Call structured extraction; model is injected automatically."""
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]  # instructor accepts plain dicts
            response_model=response_model,
            **kwargs,
        )

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """Raw text completion — no structured output. Returns the assistant message content."""
        result = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_model=None,
            **kwargs,
        )
        return result.choices[0].message.content or ""

    @property
    def token_count(self) -> int:
        """Total tokens processed; only non-zero for providers that track usage (e.g. Gemini)."""
        return getattr(self._client, "token_count", 0)

    # Expose the underlying client for callers that need raw access
    @property
    def raw(self) -> instructor.Instructor:
        return self._client


def get_llm(
    provider: Literal["ollama", "gemini"] = "ollama", model: str | None = None
) -> MosaicLLMClient:
    """Return a MosaicLLMClient for the given provider.

    Args:
        provider: 'ollama' (local) or 'gemini' (Vertex AI).
        model: Override the default model for the provider.
    """
    if provider == "ollama":
        return _get_ollama_client(model or "llama3.2:latest")
    if provider == "gemini":
        return _get_gemini_client(model or "gemini-2.5-flash")
    raise ValueError(f"Unknown LLM provider: {provider!r}. Choose 'ollama' or 'gemini'.")


def _get_ollama_client(model: str) -> MosaicLLMClient:
    # Ollama exposes an OpenAI-compatible API at localhost:11434/v1
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed. Run: uv sync") from e

    raw_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    patched = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)
    return MosaicLLMClient(patched, model)


def _get_gemini_client(model: str) -> MosaicLLMClient:
    from mosaic.llm.gemini import get_gemini_shim
    shim = get_gemini_shim(model)
    return MosaicLLMClient(shim, model)
