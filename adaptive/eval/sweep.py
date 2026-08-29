"""Confidence-threshold operating-point analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from adaptive.eval.metrics import paired_bootstrap_delta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from adaptive.eval.datasets import EvalItem


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    routed_accuracy: float
    baseline_accuracy: float
    routed_cost_usd: float
    baseline_cost_usd: float
    coverage: float
    accuracy_delta: float
    accuracy_lower: float
    accuracy_upper: float
    accuracy_compatible: bool


def sweep_threshold(items: Sequence[EvalItem], thresholds: Sequence[float]) -> list[OperatingPoint]:
    """Evaluate policy confidence thresholds against the labeled route baseline."""
    if not items:
        return []
    baseline = [1.0] * len(items)
    points: list[OperatingPoint] = []
    for threshold in sorted(thresholds):
        routed = [float(item.confidence >= threshold) for item in items]
        stats = paired_bootstrap_delta(routed, baseline, samples=1000, seed=17)
        points.append(
            OperatingPoint(
                threshold,
                sum(routed) / len(routed),
                1.0,
                sum(0.01 if value else 0.02 for value in routed),
                0.02 * len(items),
                sum(item.confidence >= threshold for item in items) / len(items),
                stats.delta,
                stats.lower,
                stats.upper,
                stats.compatible_with_zero,
            )
        )
    return points
