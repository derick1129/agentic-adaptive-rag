import pytest

from adaptive.tools.sql_validate import SQLValidationError, validate_sql


def test_rejects_mutating_and_multi_statement_sql():
    for sql in (
        "DELETE FROM public.orders",
        "UPDATE public.orders SET total = 0",
        "DROP TABLE public.orders",
        "SELECT 1; SELECT 2",
        "WITH changed AS (DELETE FROM public.orders RETURNING id) SELECT * FROM changed",
        "ATTACH DATABASE 'x' AS other",
        "PRAGMA table_info(orders)",
    ):
        with pytest.raises(SQLValidationError):
            validate_sql(sql, allowed_schemas={"public"}, max_rows=100)


def test_rejects_excessive_limit_and_unknown_schema():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM public.orders LIMIT 101", {"public"}, 100)
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM secret.orders", {"public"}, 100)


def test_adds_bounded_limit_and_server_tenant_predicate():
    result = validate_sql("SELECT id, total FROM public.orders", {"public"}, 100)
    scoped = result.with_tenant("acme")
    assert "LIMIT 100" in scoped.sql.upper()
    assert "TENANT_ID" in scoped.sql.upper()
    assert scoped.params["__tenant_id"] == "acme"


def test_explicit_cross_tenant_predicate_cannot_escape_server_scope():
    result = validate_sql(
        "SELECT * FROM public.orders WHERE tenant_id = 'globex'",
        {"public"},
        100,
    ).with_tenant("acme")
    assert "GLOBEX" in result.sql.upper()
    assert result.params["__tenant_id"] == "acme"
