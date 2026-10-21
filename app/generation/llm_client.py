"""LLM access from the backend (never from the browser): Azure OpenAI or any OpenAI-compatible API."""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings


class LLMError(Exception):
    """The model call failed (timeout, quota, outage). Details are logged, not shown to users."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    time_to_first_token: float | None = None


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> LLMResult: ...


class OpenAIChatClient:
    """Streams the completion so time-to-first-token can be measured, then returns the full text."""

    def __init__(self, client: Any, model: str, *, temperature: float, max_tokens: int):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, messages: list[dict[str, str]]) -> LLMResult:
        from openai import OpenAIError

        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        usage = None
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for event in stream:
                if event.usage is not None:
                    usage = event.usage
                for choice in event.choices:
                    delta = choice.delta.content if choice.delta else None
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        parts.append(delta)
        except OpenAIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        return LLMResult(
            text="".join(parts),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            time_to_first_token=first_token_at - started if first_token_at is not None else None,
        )


def create_llm_client(settings: Settings) -> LLMClient:
    from openai import AsyncAzureOpenAI, AsyncOpenAI

    if settings.llm_provider == "azure_openai":
        api_key = settings.azure_openai_api_key.get_secret_value() if settings.azure_openai_api_key else ""
        if api_key:
            client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=api_key,
                api_version=settings.azure_openai_api_version,
                timeout=settings.llm_timeout_seconds,
            )
        else:
            # No key: authenticate with Entra ID (managed identity in Kubernetes, az login locally).
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                azure_ad_token_provider=get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
                ),
                api_version=settings.azure_openai_api_version,
                timeout=settings.llm_timeout_seconds,
            )
        model = settings.azure_openai_deployment
    else:
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else "not-needed"
        client = AsyncOpenAI(
            base_url=settings.openai_base_url or None, api_key=api_key, timeout=settings.llm_timeout_seconds
        )
        model = settings.openai_model

    return OpenAIChatClient(client, model, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens)
