from engine.auth.static_token import StaticTokenAuthenticator
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


def test_unknown_or_missing_token_degrades_to_public():
    a = _auth()
    assert a.authenticate("nope") == Principal("anonymous", "public")
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


def test_empty_map_degrades_everything_to_public():
    a = StaticTokenAuthenticator({})
    assert a.authenticate("anything") == Principal("anonymous", "public")
    assert a.roles == set()
