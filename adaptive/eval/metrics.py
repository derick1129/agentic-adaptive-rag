"""Pure, deterministic metrics used by offline and trace-based evaluation."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_MEANINGFUL_TOKEN_LENGTH = 2


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    return len(set(ranked_ids[: max(0, k)]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(
    ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str], k: int
) -> float:
    top = list(ranked_ids[: max(0, k)])
    return len(set(top) & set(relevant_ids)) / len(top) if top else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str]) -> float:
    relevant = set(relevant_ids)
    return next((1.0 / rank for rank, value in enumerate(ranked_ids, 1) if value in relevant), 0.0)


def ndcg(
    ranked_ids: Sequence[str], relevant_ids: set[str] | Sequence[str], k: int | None = None
) -> float:
    relevant = set(relevant_ids)
    ranked = list(ranked_ids if k is None else ranked_ids[: max(0, k)])
    dcg = sum(
        (1.0 / math.log2(rank + 1)) for rank, value in enumerate(ranked, 1) if value in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), len(ranked)) + 1))
    return dcg / ideal if ideal else 0.0


def reranker_lift(before: float, after: float) -> float:
    return after - before


def answer_correctness(answer: str, gold_answer: str) -> float:
    expected, actual = _tokens(gold_answer), _tokens(answer)
    return 1.0 if expected and expected.issubset(actual) else 0.0


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    context_tokens = set().union(*(_tokens(context) for context in contexts)) if contexts else set()
    sentences = [sentence for sentence in re.split(r"[.!?]+", answer) if sentence.strip()]
    if not sentences or not context_tokens:
        return 0.0
    supported = sum(
        bool(
            {token for token in _tokens(sentence) if len(token) > _MIN_MEANINGFUL_TOKEN_LENGTH}
            & context_tokens
        )
        for sentence in sentences
    )
    return supported / len(sentences)


def citation_completeness(
    citations: Sequence[dict[str, Any] | str], expected: set[str] | Sequence[str]
) -> float:
    found = {
        citation if isinstance(citation, str) else citation.get("chunk_id")
        for citation in citations
    }
    required = set(expected)
    return len(found & required) / len(required) if required else 1.0


def refusal_quality(refused: bool, should_refuse: bool) -> float:
    return float(refused == should_refuse)


def sql_result_set_accuracy(
    actual: Sequence[dict[str, Any]], expected: Sequence[dict[str, Any]]
) -> float:
    def canonical(rows: Sequence[dict[str, Any]]) -> list[str]:
        return sorted(json_row(row) for row in rows)

    return float(canonical(actual) == canonical(expected))


def json_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)


def route_confusion_matrix(
    expected: Sequence[str], actual: Sequence[str]
) -> dict[str, dict[str, int]]:
    labels = sorted(set(expected) | set(actual))
    matrix = {label: dict.fromkeys(labels, 0) for label in labels}
    for gold, prediction in zip(expected, actual, strict=False):
        matrix.setdefault(gold, dict.fromkeys(labels, 0))
        matrix[gold].setdefault(prediction, 0)
        matrix[gold][prediction] += 1
    return matrix


@dataclass(frozen=True)
class BootstrapDelta:
    delta: float
    lower: float
    upper: float
    samples: int
    seed: int
    confidence: float = 0.95

    @property
    def compatible_with_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    @property
    def confidence_interval(self) -> tuple[float, float]:
        return self.lower, self.upper


def paired_bootstrap_delta(
    routed: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapDelta:
    if len(routed) != len(baseline) or not routed:
        raise ValueError(  # noqa: TRY003
            "paired bootstrap requires equally sized, non-empty samples"
        )
    rng = random.Random(seed)  # noqa: S311 - seeded pseudo-randomness is required for reproducibility.
    differences = [float(left) - float(right) for left, right in zip(routed, baseline, strict=True)]
    deltas = [
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(samples)
    ]
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0

    def percentile(fraction: float) -> float:
        position = (len(deltas) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return deltas[low]
        return deltas[low] + (deltas[high] - deltas[low]) * (position - low)

    return BootstrapDelta(
        sum(differences) / len(differences),
        percentile(alpha),
        percentile(1 - alpha),
        samples,
        seed,
        confidence,
    )


def bootstrap_confidence_interval(
    values: Sequence[float], *, samples: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a seeded percentile interval for a mean."""
    result = paired_bootstrap_delta(
        values, [0.0] * len(values), samples=samples, seed=seed, confidence=confidence
    )
    return result.lower, result.upper
