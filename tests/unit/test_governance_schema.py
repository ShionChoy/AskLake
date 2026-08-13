from engine.governance.policy import Policy, RolePolicy
from engine.governance.schema import governed_semantic_layer
from engine.semantic.semantic_model import ColumnDef, FewShot, SemanticLayer, TableDef


def test_llm_schema_context_omits_denied_and_masked_columns():
    layer = SemanticLayer(
        tables=(
            TableDef(
                "people",
                columns=(ColumnDef("name"), ColumnDef("birthYear"), ColumnDef("secret")),
            ),
        ),
        synonyms={"birth": "birthYear", "person": "name"},
        few_shots=(FewShot("birth", "SELECT birthYear FROM people"),),
    )
    policy = Policy(
        version=2,
        roles=("public",),
        role_rules={
            "public": RolePolicy(
                tables=("people",),
                columns={"people.birthYear": "mask", "people.secret": "deny"},
            )
        },
    )
    filtered = governed_semantic_layer(layer, policy, "public", available_tables={"people"})
    assert [column.name for column in filtered.tables[0].columns] == ["name"]
    assert filtered.synonyms == {"person": "name"}
    assert filtered.few_shots == ()
