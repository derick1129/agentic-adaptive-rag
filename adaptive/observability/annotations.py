"""Phoenix-compatible trace annotations with serving-safe failure isolation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal
from urllib import request

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class Annotation(BaseModel):
    trace_id: str
    evaluator_name: str
    evaluator_version: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    item_id: str
    label_source: Literal["human", "reference", "model-assisted"]
    span_id: str | None = None


class PhoenixAnnotationExporter:
    """Minimal JSON HTTP exporter; Phoenix credentials are optional."""

    def __init__(self, endpoint: str, timeout_seconds: float = 2.0) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def __call__(self, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(  # noqa: S310
            self.endpoint,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds):  # noqa: S310
            pass


class AnnotationWriter:
    def __init__(self, exporter: Callable[[dict[str, object]], None] | None = None) -> None:
        self.exporter = exporter
        self.failures = 0

    def write(self, annotation: Annotation) -> None:
        if self.exporter is None:
            return
        try:
            self.exporter(annotation.model_dump())
        except Exception:
            self.failures += 1
            logger.exception("Phoenix annotation export failed; serving is unaffected")

    def evaluate(
        self, evaluator: Callable[..., Annotation], *args: object, **kwargs: object
    ) -> Annotation | None:
        try:
            annotation = evaluator(*args, **kwargs)
        except Exception:
            self.failures += 1
            logger.exception("Evaluator failed; serving is unaffected")
            return None
        self.write(annotation)
        return annotation
