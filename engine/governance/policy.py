from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from engine.governance.sql import SqlGuardrails, SqlPolicyError, analyze_read_query
from engine.ports.storage import QueryResult, StorageBackend

MASK = "***"
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_ACTIONS = frozenset({"ask", "raw_sql", "graph", "export"})
_HANDLINGS = frozenset({"allow", "mask", "deny"})


class GovernanceError(Exception):
    """A request was rejected by an enforceable governance rule."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "policy_denied",
        status_code: int = 403,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class GovernanceConfigurationError(ValueError):
    """Governance configuration is invalid or cannot be safely materialized."""


@dataclass(frozen=True)
class RowFilter:
    column: str
    deny_values: tuple[str, ...]


@dataclass(frozen=True)
class Classification:
    confidentiality: str = "public_source"
    license: str = ""
    privacy: str = "none"
    integrity: str = "source_fact"
    content: str = "general"


@dataclass(frozen=True)
class TablePolicy:
    classification: Classification = Classification()
    columns: dict[str, Classification] = field(default_factory=dict)
    required: bool = False


@dataclass(frozen=True)
class GraphRelationPolicy:
    classification: Classification = Classification()
    citation_required: bool = True


@dataclass(frozen=True)
class RolePolicy:
    actions: frozenset[str] = frozenset()
    tables: tuple[str, ...] = ()
    columns: dict[str, str] = field(default_factory=dict)
    row_security: dict[str, str] = field(default_factory=dict)
    graph_relations: tuple[str, ...] = ()
    max_rows: int | None = None
    max_graph_triples: int = 100


@dataclass(frozen=True)
class Policy:
    """Dataset governance policy with legacy fields kept for adapter compatibility.

    Version-2 policies are deny-by-default and use role/resource labels. The legacy PII and
    row-filter fields remain supported so the small, generic governance adapter stays usable by
    older integrations; the production entrypoint uses ``role_rules`` and ``table_policies``.
    """

    version: int = 1
    default_effect: str = "deny"
    anonymous_role: str = "public"
    roles: tuple[str, ...] = ()
    role_rules: dict[str, RolePolicy] = field(default_factory=dict)
    table_policies: dict[str, TablePolicy] = field(default_factory=dict)
    graph_relations: dict[str, GraphRelationPolicy] = field(default_factory=dict)
    graph_default_effect: str = "deny"
    license_notices: dict[str, str] = field(default_factory=dict)
    guardrails: SqlGuardrails = SqlGuardrails()

    # Backward-compatible policy surface.
    pii_columns: tuple[str, ...] = ()
    mask_roles: tuple[str, ...] = ()
    row_filters: dict[str, tuple[RowFilter, ...]] = field(default_factory=dict)
    row_security: dict[str, dict[str, str]] = field(default_factory=dict)
    require_limit: bool = False
    forbid_writes: bool = True

    def require_role(self, role: str) -> None:
        if self.roles and role not in self.roles:
            raise GovernanceError("role is not recognized", code="unknown_role")

    def role(self, role: str) -> RolePolicy:
        self.require_role(role)
        return self.role_rules.get(role, RolePolicy())

    def allows_action(self, role: str, action: str) -> bool:
        self.require_role(role)
        if action not in _ACTIONS:
            return False
        if not self.role_rules:
            return True
        return action in self.role(role).actions

    def tables_for(self, role: str, available: set[str] | frozenset[str]) -> frozenset[str]:
        rule = self.role(role)
        if not self.role_rules or "*" in rule.tables:
            return frozenset(available)
        available_folded = {name.casefold(): name for name in available}
        return frozenset(
            available_folded[name.casefold()]
            for name in rule.tables
            if name.casefold() in available_folded
        )

    def column_handling(self, role: str, table: str, column: str) -> str:
        rule = self.role(role)
        for key in (f"{table}.{column}", f"*.{column}"):
            if key in rule.columns:
                return rule.columns[key]
        if role in self.mask_roles and column in self.pii_columns:
            return "mask"
        return "allow"

    def row_predicate(self, role: str, table: str) -> str | None:
        rule = self.role(role)
        return rule.row_security.get(table) or self.row_security.get(role, {}).get(table)

    def max_rows_for(self, role: str) -> int:
        configured = self.role(role).max_rows
        return configured if configured is not None else self.guardrails.max_rows

    def graph_relations_for(self, role: str) -> frozenset[str]:
        rule = self.role(role)
        available = frozenset(self.graph_relations)
        if not self.role_rules or "*" in rule.graph_relations:
            return available
        return frozenset(r for r in rule.graph_relations if r in available)

    def max_graph_triples_for(self, role: str) -> int:
        return self.role(role).max_graph_triples

    def obligations_for(self, tables: tuple[str, ...] = ()) -> dict[str, Any]:
        licenses: set[str] = set()
        for table_name in tables:
            table = self.table_policies.get(table_name)
            if table is None:
                continue
            if table.classification.license:
                licenses.add(table.classification.license)
            licenses.update(labels.license for labels in table.columns.values() if labels.license)
        sorted_licenses = sorted(licenses)
        return {
            "policy_version": self.version,
            "licenses": sorted_licenses,
            "notices": [
                self.license_notices[x] for x in sorted_licenses if x in self.license_notices
            ],
        }


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise GovernanceConfigurationError(f"{where} must be a mapping")
    return value


def _only_keys(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise GovernanceConfigurationError(f"unknown {where} keys: {unknown}")


def _positive_int(value: Any, where: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GovernanceConfigurationError(f"{where} must be a positive integer") from exc
    if parsed <= 0:
        raise GovernanceConfigurationError(f"{where} must be a positive integer")
    return parsed


def _classification(value: Any, where: str) -> Classification:
    data = _mapping(value, where)
    _only_keys(
        data,
        {"confidentiality", "license", "privacy", "integrity", "content"},
        where,
    )
    return Classification(**{key: str(val) for key, val in data.items()})


def _load_roles(
    data: dict[str, Any], version: int
) -> tuple[tuple[str, ...], dict[str, RolePolicy], tuple[str, ...]]:
    rules: dict[str, RolePolicy] = {}
    mask_roles: list[str] = []
    for role, raw in data.items():
        if not _SAFE_NAME.fullmatch(str(role)):
            raise GovernanceConfigurationError(f"unsafe role name: {role!r}")
        cfg = _mapping(raw, f"roles.{role}")
        _only_keys(
            cfg,
            {
                "actions",
                "tables",
                "columns",
                "row_security",
                "graph_relations",
                "max_rows",
                "max_graph_triples",
                "pii",
            },
            f"roles.{role}",
        )
        if cfg.get("pii") == "mask":
            mask_roles.append(str(role))
        default_actions = _ACTIONS - {"export"} if version == 1 else ()
        default_resources = ("*",) if version == 1 else ()
        actions = frozenset(str(x) for x in cfg.get("actions", default_actions))
        invalid_actions = sorted(actions - _ACTIONS)
        if invalid_actions:
            raise GovernanceConfigurationError(
                f"roles.{role}.actions contains unknown actions: {invalid_actions}"
            )
        tables = tuple(str(x) for x in cfg.get("tables", default_resources))
        columns = {str(k): str(v) for k, v in _mapping(cfg.get("columns"), "columns").items()}
        invalid_handling = sorted({v for v in columns.values() if v not in _HANDLINGS})
        if invalid_handling:
            raise GovernanceConfigurationError(
                f"roles.{role}.columns has invalid handling values: {invalid_handling}"
            )
        row_security = {
            str(k): str(v) for k, v in _mapping(cfg.get("row_security"), "row_security").items()
        }
        graph_relations = tuple(str(x) for x in cfg.get("graph_relations", default_resources))
        max_rows = (
            _positive_int(cfg["max_rows"], f"roles.{role}.max_rows") if "max_rows" in cfg else None
        )
        max_graph = _positive_int(
            cfg.get("max_graph_triples", 100), f"roles.{role}.max_graph_triples"
        )
        rules[str(role)] = RolePolicy(
            actions=actions,
            tables=tables,
            columns=columns,
            row_security=row_security,
            graph_relations=graph_relations,
            max_rows=max_rows,
            max_graph_triples=max_graph,
        )
    return tuple(data), rules, tuple(mask_roles)


def _load_tables(data: dict[str, Any]) -> dict[str, TablePolicy]:
    tables: dict[str, TablePolicy] = {}
    for name, raw in data.items():
        if not _SAFE_NAME.fullmatch(str(name)):
            raise GovernanceConfigurationError(f"unsafe table name: {name!r}")
        cfg = _mapping(raw, f"data.tables.{name}")
        _only_keys(cfg, {"classification", "columns", "required"}, f"data.tables.{name}")
        columns = {
            str(column): _classification(labels, f"data.tables.{name}.columns.{column}")
            for column, labels in _mapping(
                cfg.get("columns"), f"data.tables.{name}.columns"
            ).items()
        }
        tables[str(name)] = TablePolicy(
            classification=_classification(
                cfg.get("classification"), f"data.tables.{name}.classification"
            ),
            columns=columns,
            required=bool(cfg.get("required", False)),
        )
    return tables


def _load_graph(data: dict[str, Any]) -> tuple[str, dict[str, GraphRelationPolicy]]:
    _only_keys(data, {"default_effect", "relations"}, "graph")
    default_effect = str(data.get("default_effect", "deny"))
    if default_effect not in {"allow", "deny"}:
        raise GovernanceConfigurationError("graph.default_effect must be allow or deny")
    relations: dict[str, GraphRelationPolicy] = {}
    for name, raw in _mapping(data.get("relations"), "graph.relations").items():
        cfg = _mapping(raw, f"graph.relations.{name}")
        _only_keys(cfg, {"classification", "citation_required"}, f"graph.relations.{name}")
        relations[str(name)] = GraphRelationPolicy(
            classification=_classification(
                cfg.get("classification"), f"graph.relations.{name}.classification"
            ),
            citation_required=bool(cfg.get("citation_required", True)),
        )
    return default_effect, relations


def _load_guardrails(data: dict[str, Any], legacy: dict[str, Any]) -> SqlGuardrails:
    _only_keys(
        data,
        {
            "max_length",
            "max_tables",
            "max_joins",
            "max_rows",
            "require_limit",
            "forbid_cross_join",
            "forbidden_functions",
        },
        "query_guardrails",
    )
    defaults = SqlGuardrails()
    return SqlGuardrails(
        max_length=_positive_int(data.get("max_length", defaults.max_length), "max_length"),
        max_tables=_positive_int(data.get("max_tables", defaults.max_tables), "max_tables"),
        max_joins=_positive_int(data.get("max_joins", defaults.max_joins), "max_joins"),
        max_rows=_positive_int(data.get("max_rows", defaults.max_rows), "max_rows"),
        require_limit=bool(data.get("require_limit", legacy.get("require_limit", False))),
        forbid_cross_join=bool(data.get("forbid_cross_join", True)),
        forbidden_functions=frozenset(
            str(x).upper() for x in data.get("forbidden_functions", defaults.forbidden_functions)
        ),
    )


def _validate_v2_policy(
    *,
    default_effect: str,
    role_rules: dict[str, RolePolicy],
    tables: dict[str, TablePolicy],
    graph_default: str,
    graph_relations: dict[str, GraphRelationPolicy],
    licenses: dict[str, str],
) -> None:
    if default_effect != "deny" or graph_default != "deny":
        raise GovernanceConfigurationError(
            "version-2 policies require deny as the data and graph default effect"
        )
    for role, rule in role_rules.items():
        if "*" in rule.tables or "*" in rule.graph_relations:
            raise GovernanceConfigurationError(
                f"version-2 role {role!r} must enumerate tables and graph relations"
            )
        unknown_tables = sorted(set(rule.tables) - set(tables))
        if unknown_tables:
            raise GovernanceConfigurationError(
                f"role {role!r} grants unclassified tables: {unknown_tables}"
            )
        unknown_relations = sorted(set(rule.graph_relations) - set(graph_relations))
        if unknown_relations:
            raise GovernanceConfigurationError(
                f"role {role!r} grants unclassified graph relations: {unknown_relations}"
            )
    used_licenses = {
        item.classification.license for item in tables.values() if item.classification.license
    } | {
        item.classification.license
        for item in graph_relations.values()
        if item.classification.license
    }
    used_licenses.update(
        labels.license
        for table in tables.values()
        for labels in table.columns.values()
        if labels.license
    )
    missing_notices = sorted(used_licenses - set(licenses))
    if missing_notices:
        raise GovernanceConfigurationError(
            f"classified licenses have no handling notice: {missing_notices}"
        )


def load_policy(path: str | Path) -> Policy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = _mapping(raw, "policy")
    _only_keys(
        data,
        {
            "version",
            "default_effect",
            "anonymous_role",
            "roles",
            "data",
            "query_guardrails",
            "graph",
            "licenses",
            # Version-1 compatibility.
            "pii_columns",
            "row_filters",
            "row_security",
            "cost_guardrail",
        },
        "policy",
    )
    version = int(data.get("version", 1))
    if version not in {1, 2}:
        raise GovernanceConfigurationError(f"unsupported policy version: {version}")
    default_effect = str(data.get("default_effect", "deny"))
    if default_effect not in {"allow", "deny"}:
        raise GovernanceConfigurationError("default_effect must be allow or deny")

    roles_data = _mapping(data.get("roles"), "roles")
    roles, role_rules, mask_roles = _load_roles(roles_data, version)
    if version >= 2 and not roles:
        raise GovernanceConfigurationError("version-2 policy must declare at least one role")
    anonymous_role = str(data.get("anonymous_role", "public"))
    if roles and anonymous_role not in roles:
        raise GovernanceConfigurationError("anonymous_role must reference a declared role")

    dataset = _mapping(data.get("data"), "data")
    _only_keys(dataset, {"tables"}, "data")
    tables = _load_tables(_mapping(dataset.get("tables"), "data.tables"))

    graph_default, graph_relations = _load_graph(_mapping(data.get("graph"), "graph"))
    licenses = {str(k): str(v) for k, v in _mapping(data.get("licenses"), "licenses").items()}
    if version >= 2:
        _validate_v2_policy(
            default_effect=default_effect,
            role_rules=role_rules,
            tables=tables,
            graph_default=graph_default,
            graph_relations=graph_relations,
            licenses=licenses,
        )
    legacy_guard = _mapping(data.get("cost_guardrail"), "cost_guardrail")
    _only_keys(legacy_guard, {"require_limit", "forbid_writes"}, "cost_guardrail")
    guardrails = _load_guardrails(
        _mapping(data.get("query_guardrails"), "query_guardrails"), legacy_guard
    )

    row_filters = {
        str(role): tuple(
            RowFilter(str(item["column"]), tuple(str(v) for v in item.get("deny_values", [])))
            for item in (filters or [])
        )
        for role, filters in _mapping(data.get("row_filters"), "row_filters").items()
    }
    row_security = {
        str(role): {str(table): str(predicate) for table, predicate in (tables_cfg or {}).items()}
        for role, tables_cfg in _mapping(data.get("row_security"), "row_security").items()
    }
    return Policy(
        version=version,
        default_effect=default_effect,
        anonymous_role=anonymous_role,
        roles=roles,
        role_rules=role_rules,
        table_policies=tables,
        graph_relations=graph_relations,
        graph_default_effect=graph_default,
        license_notices=licenses,
        guardrails=guardrails,
        pii_columns=tuple(str(x) for x in data.get("pii_columns", ())),
        mask_roles=mask_roles,
        row_filters=row_filters,
        row_security=row_security,
        require_limit=guardrails.require_limit,
        forbid_writes=bool(legacy_guard.get("forbid_writes", True)),
    )


class PolicyGovernance:
    """Deny-by-default authorization, AST query checks, masking, and result guardrails."""

    def __init__(self, policy: Policy, *, action: str = "raw_sql"):
        self._p = policy
        self._action = action

    @property
    def policy(self) -> Policy:
        return self._p

    @property
    def action(self) -> str:
        return self._action

    @classmethod
    def from_yaml(cls, path: str | Path, *, action: str = "raw_sql") -> PolicyGovernance:
        return cls(load_policy(path), action=action)

    def for_action(self, action: str) -> PolicyGovernance:
        return PolicyGovernance(self._p, action=action)

    def authorize(self, role: str, action: str | None = None) -> None:
        requested = action or self._action
        if not self._p.allows_action(role, requested):
            raise GovernanceError(
                f"role {role!r} is not permitted to perform {requested!r}",
                code="action_denied",
            )

    def before_query(self, sql: str, role: str) -> str:
        self.authorize(role)
        # Preserve the constructor-level legacy switch. Version-2 YAML loads the same value into
        # both fields, while direct Policy(...) users historically set only require_limit.
        guardrails = replace(self._p.guardrails, require_limit=self._p.require_limit)
        try:
            analyze_read_query(sql, guardrails)
        except SqlPolicyError as exc:
            denied = {
                "cross_join_denied",
                "external_access_denied",
                "schema_access_denied",
                "statement_not_read_only",
                "table_access_denied",
            }
            raise GovernanceError(
                str(exc),
                code=exc.code,
                status_code=403 if exc.code in denied else 400,
            ) from exc
        return sql

    def after_result(self, result: QueryResult, role: str) -> QueryResult:
        self.authorize(role)
        rows = [list(row) for row in result.rows]
        for row_filter in self._p.row_filters.get(role, ()):
            if row_filter.column in result.columns:
                index = result.columns.index(row_filter.column)
                rows = [row for row in rows if str(row[index]) not in row_filter.deny_values]
        # View-level masking is the primary enforcement. This is a second line of defense for
        # direct adapter users and legacy integrations; an alias cannot recover a value already
        # replaced with NULL by the role view.
        for index, column in enumerate(result.columns):
            if self._p.column_handling(role, "*", column) == "mask":
                for row in rows:
                    row[index] = MASK
        rows = rows[: self._p.max_rows_for(role)]
        return QueryResult(columns=list(result.columns), rows=[tuple(row) for row in rows])

    def scoped_backend(self, backend: StorageBackend, role: str) -> StorageBackend:
        from engine.lakehouse.role_scoped_backend import RoleScopedBackend

        self.authorize(role)
        return RoleScopedBackend(backend, role, policy=self._p, action=self._action)

    def response_metadata(self, role: str, sql: str | None = None) -> dict[str, Any]:
        self._p.require_role(role)
        tables: tuple[str, ...] = ()
        if sql:
            try:
                tables = analyze_read_query(sql, self._p.guardrails).tables
            except SqlPolicyError:
                pass
        return {
            "role": role,
            "action": self._action,
            "max_rows": self._p.max_rows_for(role),
            **self._p.obligations_for(tables),
        }
