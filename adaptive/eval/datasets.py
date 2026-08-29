"""Versioned, deterministic evaluation dataset loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable as IterableType


class EvalItem(BaseModel):
    """One labeled query and the evidence/route expectations for it."""

    id: str
    question: str
    gold_answer: str
    gold_depth: str
    gold_tool: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tenant: str
    expected_citations: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    expected_rows: list[dict[str, Any]] | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


def _paths(source: str | Path) -> IterableType[Path]:
    path = Path(source)
    if path.is_file():
        yield path
        return
    yield from sorted(path.glob("*.jsonl"))


def load_dataset(source: str | Path) -> list[EvalItem]:
    """Load JSONL files in stable filename/id order, rejecting malformed rows."""

    items: list[EvalItem] = []
    for path in _paths(source):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    items.append(EvalItem.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(  # noqa: TRY003
                        f"invalid evaluation item in {path}:{line_number}"
                    ) from exc
    return sorted(items, key=lambda item: item.id)
