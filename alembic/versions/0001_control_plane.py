"""Create tenant-scoped control-plane tables."""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0001_control_plane"
down_revision = None
branch_labels = None
depends_on = None


def _timestamps(table: sa.Table) -> None:
    table.append_column(
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    table.append_column(
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now())
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "tenants",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("slug", sa.String(120), unique=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("source_type", sa.String(40), nullable=False),
        sa.Column("source_uri", sa.Text()),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(500)),
        sa.Column("canonical_text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("acl", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "document_versions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("title", sa.String(500)),
        sa.Column("canonical_text", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("acl", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "version", name="uq_document_version_number"),
    )
    op.create_index(
        "uq_document_active_version",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("heading_path", sa.JSON(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("acl", sa.JSON(), nullable=False),
        sa.Column("embedding_status", sa.String(32), nullable=False),
        sa.Column("index_status", sa.String(32), nullable=False),
        sa.Column("embedding", Vector(1536)),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_chunks_tenant_document_version", "chunks", ["tenant_id", "document_id", "version"]
    )
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("current_stage", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "query_runs",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("subject_id", sa.String(255), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "semantic_cache_entries",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("acl", sa.JSON(), nullable=False),
        sa.Column("query_embedding", Vector(1536), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("model_profile", sa.String(255), nullable=False),
        sa.Column("response_mode", sa.String(64), nullable=False),
        sa.Column("referenced_document_versions", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_cache_tenant_expiry", "semantic_cache_entries", ["tenant_id", "expires_at"])
    op.create_table(
        "evaluation_annotations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "tenant_id", sa.UUID(), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("query_run_id", sa.UUID(), sa.ForeignKey("query_runs.id", ondelete="CASCADE")),
        sa.Column("evaluator", sa.String(255), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("dataset_item_id", sa.String(255), nullable=False),
        sa.Column("label_source", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    for table in (
        "evaluation_annotations",
        "semantic_cache_entries",
        "query_runs",
        "ingestion_jobs",
        "chunks",
        "document_versions",
        "documents",
        "tenants",
    ):
        op.drop_table(table)
