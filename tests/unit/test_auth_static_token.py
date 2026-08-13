import pytest

from engine.auth.static_token import AuthenticationError, StaticTokenAuthenticator, token_digest
from engine.ports.auth import Authenticator, Principal


def test_principal_is_frozen():
    p = Principal(user="alice", role="analyst")
    assert (p.user, p.role) == ("alice", "analyst")
    import dataclasses

    assert dataclasses.is_dataclass(p)
    try:
        p.role = "public"  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_principal_satisfies_authenticator_protocol():
    class A:
        def authenticate(self, credential):
            return Principal("anonymous", "public")

    assert isinstance(A(), Authenticator)


def _auth() -> StaticTokenAuthenticator:
    return StaticTokenAuthenticator(
        {"tok_a": Principal("alice", "analyst"), "tok_p": Principal("viewer", "public")}
    )


def test_known_token_maps_to_principal():
    assert _auth().authenticate("tok_a") == Principal("alice", "analyst")


def test_unknown_token_is_rejected_but_missing_token_is_anonymous():
    a = _auth()
    with pytest.raises(AuthenticationError, match="invalid"):
        a.authenticate("nope")
    assert a.authenticate(None) == Principal("anonymous", "public")
    assert a.authenticate("") == Principal("anonymous", "public")


def test_roles_property_lists_distinct_roles():
    assert _auth().roles == {"analyst", "public"}


def test_from_yaml_round_trip(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "tokens:\n  tok_a: {user: alice, role: analyst}\n  tok_p: {user: viewer, role: public}\n"
    )
    a = StaticTokenAuthenticator.from_yaml(p)
    assert a.authenticate("tok_a") == Principal("alice", "analyst")
    assert a.roles == {"analyst", "public"}


def test_version_two_config_uses_only_token_hashes(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\n"
        "credentials:\n"
        f"  - {{token_sha256: {token_digest('tok_a')}, user: alice, role: analyst}}\n"
    )
    auth = StaticTokenAuthenticator.from_yaml(p)
    assert auth.authenticate("tok_a") == Principal("alice", "analyst")
    assert "tok_a" not in repr(auth.__dict__)


def test_empty_map_allows_only_missing_credentials_as_public():
    a = StaticTokenAuthenticator({})
    with pytest.raises(AuthenticationError):
        a.authenticate("anything")
    assert a.authenticate(None) == Principal("anonymous", "public")
    assert a.roles == set()
