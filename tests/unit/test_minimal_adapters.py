from engine.governance.passthrough import PassthroughGovernance
from engine.llm.fake import FakeLLMProvider
from engine.observability.noop import NoopObservability
from engine.ports.governance import GovernanceHook
from engine.ports.llm import LLMProvider
from engine.ports.observability import Observability
from engine.ports.storage import QueryResult


def test_passthrough_governance_is_identity():
    g = PassthroughGovernance()
    assert isinstance(g, GovernanceHook)
    assert g.before_query("SELECT 1", role="analyst") == "SELECT 1"
    qr = QueryResult(["a"], [(1,)])
    assert g.after_result(qr, role="analyst") is qr


def test_noop_observability_span_and_event():
    o = NoopObservability()
    assert isinstance(o, Observability)
    with o.span("test", k="v"):
        o.event("hello", n=1)


def test_fake_llm_returns_scripted_response():
    llm = FakeLLMProvider(responses=["SELECT 42"])
    assert isinstance(llm, LLMProvider)
    assert llm.complete("anything") == "SELECT 42"


def test_fake_llm_cycles_and_records_prompts():
    llm = FakeLLMProvider(responses=["a", "b"])
    assert llm.complete("p1") == "a"
    assert llm.complete("p2") == "b"
    assert llm.complete("p3") == "a"  # cycles
    assert llm.prompts == ["p1", "p2", "p3"]
