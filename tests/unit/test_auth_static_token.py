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
