from __future__ import annotations

from adaptive.observability.annotations import Annotation, AnnotationWriter


def test_annotation_writer_records_evaluation_fields() -> None:
    written: list[dict[str, object]] = []
    writer = AnnotationWriter(exporter=written.append)
    annotation = Annotation(
        trace_id="trace-1",
        span_id="span-1",
        evaluator_name="groundedness",
        evaluator_version="1.0",
        score=0.9,
        explanation="Supported by cited evidence",
        item_id="item-7",
        label_source="reference",
    )

    assert writer.write(annotation) is None
    assert written == [annotation.model_dump()]


def test_evaluator_or_exporter_failure_is_isolated_from_serving() -> None:
    calls = 0

    def broken_exporter(_: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("Phoenix unavailable")  # noqa: TRY003

    writer = AnnotationWriter(exporter=broken_exporter)
    annotation = Annotation(
        trace_id="trace-1",
        evaluator_name="answer_correctness",
        evaluator_version="1.0",
        score=1.0,
        explanation="Correct",
        item_id="item-1",
        label_source="human",
    )

    assert writer.write(annotation) is None
    assert calls == 1
    assert writer.failures == 1
