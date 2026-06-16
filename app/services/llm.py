from __future__ import annotations

import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.config import Settings


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None

    @property
    def is_configured(self) -> bool:
        return self.settings.llm_configured

    @property
    def client(self) -> OpenAI:
        if not self.is_configured:
            raise RuntimeError("OpenRouter API key nie jest skonfigurowany")
        if self._client is None:
            self._client = OpenAI(
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
            )
        return self._client

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> LLMResponse:
        if not self.is_configured:
            raise RuntimeError("OpenRouter API key nie jest skonfigurowany")

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.openrouter_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                choice = response.choices[0].message.content or ""
                usage = response.usage
                return LLMResponse(
                    content=choice.strip(),
                    model=response.model or self.settings.openrouter_model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                )
            except (APIConnectionError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                if isinstance(exc, APIStatusError):
                    detail = exc.message
                    try:
                        body = exc.body
                        if isinstance(body, dict):
                            detail = body.get("error", {}).get("message", detail)
                    except Exception:
                        pass
                    raise RuntimeError(f"OpenRouter API error ({exc.status_code}): {detail}") from exc
                raise RuntimeError(f"OpenRouter connection error: {exc}") from exc
        raise RuntimeError(f"Błąd generowania odpowiedzi: {last_error}")
