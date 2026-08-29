"""Safe local document sources for uploads and filesystem paths."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from adaptive.ingestion.models import SourcePayload

DEFAULT_MAX_SIZE = 25 * 1024 * 1024


def _validate_size(size: int, max_size: int) -> None:
    if size < 0 or max_size < 0:
        raise ValueError("size limits must be non-negative")  # noqa: TRY003
    if size > max_size:
        raise ValueError(f"source size {size} exceeds maximum size {max_size}")  # noqa: TRY003


class LocalUploadSource:
    """An in-memory upload whose bytes have already crossed the upload boundary."""

    def __init__(
        self,
        data: bytes,
        *,
        filename: str,
        mime_type: str,
        max_size: int = DEFAULT_MAX_SIZE,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(data, bytes):
            raise TypeError("upload data must be bytes")  # noqa: TRY003
        _validate_size(len(data), max_size)
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a basename")  # noqa: TRY003
        if not mime_type:
            raise ValueError("mime_type is required")  # noqa: TRY003
        self._payload = SourcePayload(
            data=data,
            filename=filename,
            mime_type=mime_type.split(";", 1)[0].strip().lower(),
            metadata=metadata or {},
        )

    async def read(self) -> SourcePayload:
        return self._payload

    @property
    def mime_type(self) -> str:
        return self._payload.mime_type

    @property
    def filename(self) -> str:
        return self._payload.filename

    @property
    def size(self) -> int:
        return len(self._payload.data)


class FilesystemSource:
    """A regular file constrained to an explicitly configured directory."""

    def __init__(
        self,
        path: str | Path,
        *,
        allowed_root: str | Path,
        max_size: int = DEFAULT_MAX_SIZE,
        mime_type: str | None = None,
    ) -> None:
        self._path = Path(path).expanduser().resolve(strict=False)
        self._root = Path(allowed_root).expanduser().resolve(strict=True)
        try:
            self._path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("filesystem source must be within allowed root") from exc  # noqa: TRY003
        if not self._path.is_file():
            raise ValueError("filesystem source must be a regular file")  # noqa: TRY003
        self._size = self._path.stat().st_size
        _validate_size(self._size, max_size)
        self._max_size = max_size
        guessed_mime = mimetypes.guess_type(self._path.name)[0] or "text/plain"
        self._mime_type = (mime_type or guessed_mime).lower()

    async def read(self) -> SourcePayload:
        data = self._path.read_bytes()
        _validate_size(len(data), self._max_size)
        return SourcePayload(
            data=data,
            filename=self._path.name,
            mime_type=self._mime_type,
            source_uri=str(self._path),
            metadata={"allowed_root": str(self._root)},
        )

    @property
    def mime_type(self) -> str:
        return self._mime_type

    @property
    def filename(self) -> str:
        return self._path.name

    @property
    def size(self) -> int:
        return self._size
