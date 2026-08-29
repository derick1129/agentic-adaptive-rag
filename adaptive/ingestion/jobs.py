"""State transitions for ingestion jobs."""

from __future__ import annotations

from adaptive.interfaces import IngestionJob, IngestionStatus

_NEXT = {
    IngestionStatus.RECEIVED: IngestionStatus.PARSING,
    IngestionStatus.PARSING: IngestionStatus.CHUNKING,
    IngestionStatus.CHUNKING: IngestionStatus.EMBEDDING,
    IngestionStatus.EMBEDDING: IngestionStatus.INDEXING,
    IngestionStatus.INDEXING: IngestionStatus.ACTIVE,
}


def advance(job: IngestionJob) -> IngestionJob:
    next_status = _NEXT.get(job.status)
    if next_status is None:
        return job
    return job.model_copy(update={"status": next_status, "current_stage": next_status.value})


def failed(job: IngestionJob, stage: str, error_code: str, message: str) -> IngestionJob:
    safe_message = " ".join(message.split())[:500]
    return job.model_copy(
        update={
            "status": IngestionStatus.FAILED,
            "current_stage": stage,
            "attempts": job.attempts + 1,
            "error_code": error_code,
            "error_message": safe_message,
        }
    )
