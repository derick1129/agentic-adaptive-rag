"""Fallback embedding provider that tries primary then fallback."""

from __future__ import annotations

from typing import Any, Sequence

from adaptive.interfaces import EmbeddingProvider


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that tries a primary provider, then falls back to a secondary provider."""

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def embed(
        self, texts: Sequence[str], *, input_type: str | None = None
    ) -> list[list[float]]:
        try:
            result = await self.primary.embed(texts, input_type=input_type)
            # Check if the result is valid (non-empty and correct dimensions)
            if result and len(result) == len(texts):
                return result
        except Exception:
            # Log the error? We'll just fall back.
            pass
        # Fall back to the secondary provider
        return await self.fallback.embed(texts, input_type=input_type)

    @property
    def dimensions(self) -> int:
        # Return the dimensions of the primary provider, assuming they are the same
        return self.primary.dimensions