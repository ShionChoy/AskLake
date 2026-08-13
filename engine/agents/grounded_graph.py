from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from engine.agents.agentic_graph import CORRECTION_SYSTEM, CORRECTION_TEMPLATE
from engine.agents.critic import assess, is_ranking_question, select_consistent
from engine.agents.graph import PROMPT_TEMPLATE, SYSTEM_PROMPT, extract_sql
from engine.ports.agentgraph import GraphState
from engine.ports.llm import LLMProvider
from engine.ports.schema import SchemaProvider
from engine.ports.storage import QueryResult
from engine.semantic.value_index import ValueIndex, format_hints

PLAN_SYSTEM = (
    "You are a senior analytics engineer. Produce a short query plan (tables, joins, "
    "filters, group-by, order-by key, limit). Be concise; do not write SQL."
)
PLAN_TEMPLATE = """Database schema:
{schema}

Value hints (real stored values to filter on):
{hints}

Question: {question}

Write a short numbered query plan (no SQL)."""


class GroundedState(GraphState, total=False):
    """Additive sub-class of the port's GraphState — keeps engine/ports/agentgraph.py unchanged."""

    value_hints: str
    has_entity: bool
    difficulty: str
    plan: str
    candidates: list[str]
    critique: str


def build_grounded_sql_graph(
    llm: LLMProvider,
    schema_provider: SchemaProvider,
    executor: Callable[[str], QueryResult],
    *,
    value_index: ValueIndex | None = None,
    k_candidates: int = 3,
    max_retries: int = 2,
    use_plan: bool = True,
    use_critic: bool = True,
):
    """Grounded, correctness-aware NL->SQL graph (additive sibling of build_agentic_sql_graph).

    link -> classify -> (hard & use_plan: plan ->) write(K) -> validate_select -> critic
        -> (correct -> validate_select ...) -> END. Bounded by attempts < max_retries.
    `use_plan`/`use_critic`/`k_candidates` are ablation knobs (a later plan reuses them)."""

    def _write_prompt(state: GroundedState) -> str:
        prompt = PROMPT_TEMPLATE.format(
            schema=state.get("schema_context", ""), question=state["question"]
        )
        hints = state.get("value_hints", "")
        if hints:
            prompt = f"{prompt}\n\nValue hints (use these exact stored values):\n{hints}"
        plan = state.get("plan", "")
        if plan:
            prompt = f"{prompt}\n\nQuery plan:\n{plan}"
        return prompt

    def link(state: GroundedState) -> GroundedState:
        question = state["question"]
        ctx = schema_provider.schema_context(question)
        hints = value_index.link(question) if value_index is not None else []
        return {
            "schema_context": ctx,
            "value_hints": format_hints(hints),
            "has_entity": any(h.mode == "entity" for h in hints),
            "attempts": 0,
        }

    def classify(state: GroundedState) -> GroundedState:
        is_hard = is_ranking_question(state["question"]) or state.get("has_entity", False)
        return {"difficulty": "hard" if is_hard else "simple"}

    def plan(state: GroundedState) -> GroundedState:
        prompt = PLAN_TEMPLATE.format(
            schema=state.get("schema_context", ""),
            hints=state.get("value_hints", "") or "(none)",
            question=state["question"],
        )
        return {"plan": llm.complete(prompt, system=PLAN_SYSTEM).strip()}

    def write(state: GroundedState) -> GroundedState:
        prompt = _write_prompt(state)
        k = 1 if state.get("difficulty") == "simple" else max(1, k_candidates)
        cands = [extract_sql(llm.complete(prompt, system=SYSTEM_PROMPT)) for _ in range(k)]
        return {"candidates": cands}

    def validate_select(state: GroundedState) -> GroundedState:
        executed: list[tuple[str, QueryResult]] = []
        last_error = ""
        cands = state.get("candidates") or [""]
        for sql in cands:
            try:
                executed.append((sql, executor(sql)))
            except Exception as exc:  # noqa: BLE001 - feeds the correction loop
                if getattr(exc, "status_code", None) == 403:
                    raise
                last_error = str(exc)
        if not executed:
            return {"sql": cands[0], "result": None, "error": last_error}
        sql, result = select_consistent(executed)
        return {"sql": sql, "result": result, "error": ""}

    def critic(state: GroundedState) -> GroundedState:
        if not use_critic:
            return {"critique": ""}
        c = assess(
            state["question"],
            state.get("sql", ""),
            state.get("result"),
            difficulty=state.get("difficulty", ""),
        )
        return {"critique": "" if c.ok else "; ".join(c.reasons)}

    def correct(state: GroundedState) -> GroundedState:
        diagnosis = (
            state.get("critique") or state.get("error") or "the result did not answer the question"
        )
        prompt = CORRECTION_TEMPLATE.format(
            schema=state.get("schema_context", ""),
            question=state["question"],
            sql=state.get("sql", ""),
            error=diagnosis,
        )
        raw = llm.complete(prompt, system=CORRECTION_SYSTEM)
        return {"candidates": [extract_sql(raw)], "attempts": state.get("attempts", 0) + 1}

    def route_difficulty(state: GroundedState) -> str:
        return "plan" if (use_plan and state.get("difficulty") == "hard") else "write"

    def route_after_critic(state: GroundedState) -> str:
        needs_fix = bool(state.get("error")) or bool(state.get("critique"))
        if needs_fix and state.get("attempts", 0) < max_retries:
            return "correct"
        return "done"

    g = StateGraph(GroundedState)
    g.add_node("link", link)
    g.add_node("classify", classify)
    g.add_node("plan", plan)
    g.add_node("write", write)
    g.add_node("validate_select", validate_select)
    g.add_node("critic", critic)
    g.add_node("correct", correct)
    g.add_edge(START, "link")
    g.add_edge("link", "classify")
    g.add_conditional_edges("classify", route_difficulty, {"plan": "plan", "write": "write"})
    g.add_edge("plan", "write")
    g.add_edge("write", "validate_select")
    g.add_edge("validate_select", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"correct": "correct", "done": END})
    g.add_edge("correct", "validate_select")
    return g.compile()
