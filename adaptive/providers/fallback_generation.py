"""Fallback generation provider that tries primary then fallback."""

from __future__ import annotations

from typing import Any, Type

from adaptive.interfaces import Generator


class FallbackGenerationProvider(Generator):
    """Generation provider that tries a primary provider, then falls back to a secondary provider."""

    def __init__(
        self,
        primary: Generator,
        fallback: Generator,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def generate(self, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
        try:
            return await self.primary.generate(prompt, max_tokens, temperature)
        except Exception:
            # Fall back to the secondary provider
            return await self.fallback.generate(prompt, max_tokens, temperature)

    async def generate_structured(
        self, prompt: str, schema: Type[BaseModel], max_tokens: int
    ) -> BaseModel:
        try:
            return await self.primary.generate_structured(prompt, schema, max_tokens)
        except Exception:
            # Fall back to the secondary provider
            return await self.fallback.generate_structured(prompt, schema, max_tokens)