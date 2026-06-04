import pytest

from eval.hermetic import run_hermetic_comparison


def test_agentic_beats_baseline_on_mini_set():
    baseline, agentic = run_hermetic_comparison()
    assert baseline.n == agentic.n == 3
    assert agentic.execution_accuracy > baseline.execution_accuracy
    assert agentic.execution_accuracy == 1.0
    assert baseline.execution_accuracy == pytest.approx(2 / 3)
    assert agentic.avg_attempts > 0  # at least one self-correction happened
