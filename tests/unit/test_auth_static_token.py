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


def test_plaintext_yaml_is_rejected(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "tokens:\n  tok_a: {user: alice, role: analyst}\n  tok_p: {user: viewer, role: public}\n"
    )
    with pytest.raises(AuthenticationError, match="version 2"):
        StaticTokenAuthenticator.from_yaml(p)


def test_version_two_config_uses_only_token_hashes(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\n"
        "credentials:\n"
        f"  - {{id: analyst-primary, token_sha256: {token_digest('tok_a')}, "
        "user: alice, role: analyst}\n"
    )
    auth = StaticTokenAuthenticator.from_yaml(p)
    assert auth.authenticate("tok_a") == Principal("alice", "analyst", "analyst-primary")
    assert "tok_a" not in repr(auth.__dict__)


def test_expired_and_disabled_credentials_are_rejected(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\ncredentials:\n"
        f"  - {{id: expired, token_sha256: {token_digest('old')}, user: alice, "
        "role: analyst, expires_at: '2020-01-01T00:00:00Z'}\n"
        f"  - {{id: disabled, token_sha256: {token_digest('off')}, user: bob, "
        "role: steward, disabled: true}\n"
    )
    auth = StaticTokenAuthenticator.from_yaml(p)
    with pytest.raises(AuthenticationError, match="expired"):
        auth.authenticate("old")
    with pytest.raises(AuthenticationError, match="disabled"):
        auth.authenticate("off")


def test_v2_credential_requires_lifecycle_id(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\ncredentials:\n"
        f"  - {{token_sha256: {token_digest('tok_a')}, user: alice, role: analyst}}\n"
    )
    with pytest.raises(AuthenticationError, match="id is required"):
        StaticTokenAuthenticator.from_yaml(p)


def test_v2_boolean_controls_must_be_real_booleans(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text("version: 2\nallow_anonymous: 'false'\ncredentials: []\n")
    with pytest.raises(AuthenticationError, match="true or false"):
        StaticTokenAuthenticator.from_yaml(p)


def test_fail_closed_config_requires_an_active_credential(tmp_path):
    p = tmp_path / "auth.yaml"
    p.write_text(
        "version: 2\nallow_anonymous: false\ncredentials:\n"
        f"  - {{id: expired, token_sha256: {token_digest('old')}, user: alice, "
        "role: analyst, expires_at: '2020-01-01T00:00:00Z'}\n"
    )
    with pytest.raises(AuthenticationError, match="no active credentials"):
        StaticTokenAuthenticator.from_yaml(p)


def test_empty_map_allows_only_missing_credentials_as_public():
    a = StaticTokenAuthenticator({})
    with pytest.raises(AuthenticationError):
        a.authenticate("anything")
    assert a.authenticate(None) == Principal("anonymous", "public")
    assert a.roles == set()
