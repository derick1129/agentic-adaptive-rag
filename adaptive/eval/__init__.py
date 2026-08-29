"""Deterministic offline evaluation for adaptive RAG."""

from adaptive.eval.datasets import EvalItem, load_dataset
from adaptive.eval.metrics import paired_bootstrap_delta
from adaptive.eval.runner import EvaluationReport, evaluate_run

__all__ = ["EvalItem", "EvaluationReport", "evaluate_run", "load_dataset", "paired_bootstrap_delta"]
