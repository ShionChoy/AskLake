import inspect

import engine.governance.passthrough as passthrough
import engine.retrieval.agentic_sql_path as agentic
import engine.retrieval.grounded_sql_path as grounded
import engine.retrieval.sql_path as sql_path


def test_prior_retrieval_paths_do_not_import_auth_stack():
    # additive guarantee: the engine SQL paths must not depend on the new access-control stack
    for mod in (sql_path, agentic, grounded, passthrough):
        src = inspect.getsource(mod)
        assert "role_scoped_backend" not in src
        assert "engine.auth" not in src
        assert "engine.governance.views" not in src


def test_passthrough_governance_unchanged_signature():
    from engine.governance.passthrough import PassthroughGovernance

    g = PassthroughGovernance()
    assert g.before_query("SELECT 1", role="public") == "SELECT 1"
