import numpy as np
import pytest

from fel_cl import evaluate_on_past_tasks, forgetting_metrics


def test_evaluate_on_past_tasks_skips_current_cycle_and_uses_injected_metric():
    task_val_data = {j: (np.full((5, 1), j), np.zeros((5, 1))) for j in range(3)}
    seen = []

    def metric_fn(model, x, y):
        seen.append(int(x[0, 0]))
        return float(x[0, 0]) * 10

    result = evaluate_on_past_tasks(object(), task_val_data, current_cycle=2, metric_fn=metric_fn)

    assert result == {0: 0.0, 1: 10.0}
    assert seen == [0, 1]


def test_forgetting_metrics_lower_is_better():
    nan = np.nan
    m = np.array([[0.1, nan, nan], [0.3, 0.1, nan], [0.4, 0.3, 0.1]])
    out = forgetting_metrics(m)
    assert out["final_average"] == pytest.approx((0.4 + 0.3 + 0.1) / 3)
    # task 0: final 0.4 vs best earlier 0.1 -> 0.3; task 1: 0.3 vs 0.1 -> 0.2
    assert out["average_forgetting"] == pytest.approx(0.25)
    assert out["backward_transfer"] == pytest.approx(-0.25)


def test_forgetting_metrics_single_task():
    out = forgetting_metrics(np.array([[0.2]]))
    assert out["average_forgetting"] is None and out["backward_transfer"] is None
