"""OpenTelemetry tracing, metrics, and Phoenix-compatible annotations."""

from adaptive.observability.annotations import (
    Annotation,
    AnnotationWriter,
    PhoenixAnnotationExporter,
)
from adaptive.observability.attributes import TelemetryConfig, normalize_attributes
from adaptive.observability.tracing import QueryTrace, TraceManager, configure_telemetry

__all__ = [
    "Annotation",
    "AnnotationWriter",
    "PhoenixAnnotationExporter",
    "QueryTrace",
    "TelemetryConfig",
    "TraceManager",
    "configure_telemetry",
    "normalize_attributes",
]
