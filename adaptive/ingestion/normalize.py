"""Canonical text and metadata normalization."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from adaptive.ingestion.models import CanonicalDocument, ParsedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable


def _normalize_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_nonempty(values: Iterable[str | None]) -> str | None:
    return next((value.strip() for value in values if value and value.strip()), None)


def normalize(parsed: ParsedDocument, *, acl: Iterable[str] = ()) -> CanonicalDocument:
    canonical_text = _normalize_lines(parsed.text)
    title = _first_nonempty([parsed.title, *(heading.get("text") for heading in parsed.headings)])
    metadata = dict(parsed.metadata)
    if parsed.pages:
        metadata.setdefault("page_count", len(parsed.pages))
    if parsed.headings:
        metadata.setdefault("heading_count", len(parsed.headings))
    if parsed.tables:
        metadata.setdefault("table_count", len(parsed.tables))
    content_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return CanonicalDocument(
        content_hash=content_hash,
        title=title,
        canonical_text=canonical_text,
        metadata=metadata,
        acl=frozenset(acl),
        pages=parsed.pages,
        headings=parsed.headings,
        tables=parsed.tables,
    )
