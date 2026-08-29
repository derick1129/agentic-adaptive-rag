"""AST-level validation and scoping for read-only SQL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class SQLValidationError(ValueError):
    """Raised when a generated SQL statement is unsafe or outside policy."""


@dataclass(frozen=True)
class ValidatedSQL:
    sql: str
    ast: exp.Expression
    max_rows: int
    params: dict[str, Any]

    def with_tenant(self, tenant_id: str) -> "ValidatedSQL":
        if not tenant_id or any(char in tenant_id for char in ("\x00", "\n", "\r")):
            raise SQLValidationError("tenant_id is invalid")

        predicate: exp.Expression | None = None
        tables = list(self.ast.find_all(exp.Table))
        for table in tables:
            # A CTE reference is not a physical table and cannot be scoped here.
            if table.name in {cte.alias_or_name for cte in self.ast.find_all(exp.CTE)}:
                continue
            qualifier = table.alias_or_name or table.name
            column = exp.column("tenant_id", table=qualifier)
            current = exp.EQ(this=column, expression=exp.Literal.string(tenant_id))
            predicate = (
                current if predicate is None else exp.And(this=predicate, expression=current)
            )

        if predicate is None:
            raise SQLValidationError("query must reference a tenant-scoped table")

        select = self.ast.find(exp.Select)
        if select is None:
            raise SQLValidationError("only SELECT statements are allowed")
        existing = select.args.get("where")
        select.set(
            "where",
            exp.Where(
                this=predicate
                if existing is None
                else exp.And(this=existing.this, expression=predicate)
            ),
        )
        return ValidatedSQL(
            sql=self.ast.sql(dialect="postgres"),
            ast=self.ast,
            max_rows=self.max_rows,
            params={"__tenant_id": tenant_id},
        )


_FORBIDDEN = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Alter,
    exp.Drop,
    exp.Merge,
    exp.Command,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)


def validate_sql(
    sql: str,
    allowed_schemas: set[str] | frozenset[str] = frozenset({"public"}),
    max_rows: int = 1000,
) -> ValidatedSQL:
    """Parse one bounded SELECT and reject all write/control statements."""
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty")
    if max_rows < 1:
        raise SQLValidationError("max_rows must be positive")
    try:
        statements = sqlglot.parse(sql, read="postgres")
    except ParseError as exc:
        raise SQLValidationError("SQL could not be parsed") from exc
    if len(statements) != 1:
        raise SQLValidationError("exactly one SQL statement is required")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SQLValidationError("only SELECT statements are allowed")
    if any(statement.find(node_type) is not None for node_type in _FORBIDDEN):
        raise SQLValidationError("read-only SELECT required")
    if any(isinstance(node, (exp.Into, exp.Lock)) for node in statement.walk()):
        raise SQLValidationError("locking and INTO clauses are not allowed")

    allowed = set(allowed_schemas)
    for table in statement.find_all(exp.Table):
        if table.catalog or (table.db and table.db not in allowed):
            raise SQLValidationError("table schema is not allowlisted")
        if not table.name:
            raise SQLValidationError("table name is required")

    limit = statement.args.get("limit")
    if limit is not None:
        expression = limit.expression
        if not isinstance(expression, exp.Literal) or not expression.is_int:
            raise SQLValidationError("LIMIT must be a positive integer")
        if int(expression.this) > max_rows:
            raise SQLValidationError("LIMIT exceeds configured maximum")
    else:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return ValidatedSQL(
        sql=statement.sql(dialect="postgres"),
        ast=statement,
        max_rows=max_rows,
        params={},
    )


def validate_and_scope_sql(
    sql: str,
    tenant_id: str,
    allowed_schemas: set[str] | frozenset[str] = frozenset({"public"}),
    max_rows: int = 1000,
) -> ValidatedSQL:
    """Convenience boundary used by SQL tools and callers outside the tool class."""
    return validate_sql(sql, allowed_schemas, max_rows).with_tenant(tenant_id)
