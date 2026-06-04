from eval.semantic_hermetic import run_semantic_comparison


def test_semantic_layer_beats_raw_schema():
    raw, semantic = run_semantic_comparison()
    assert raw.n == semantic.n == 1
    assert raw.execution_accuracy == 0.0  # bare schema keeps guessing a non-existent column
    assert semantic.execution_accuracy == 1.0  # synonym grounding fixes it
    assert semantic.execution_accuracy > raw.execution_accuracy
