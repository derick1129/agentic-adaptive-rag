"""SQLAlchemy engine and session helpers for the control plane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.orm import Session

from adaptive.config import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create an engine without opening a connection until it is used."""
    url = database_url or get_settings().database_url
    options: dict[str, object] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


engine = create_database_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """Yield a request-scoped session and always close it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
