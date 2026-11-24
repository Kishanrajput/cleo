import numpy as np
import pytest

from l3kit.replay import ReplayBuffer


def test_add_task_selects_up_to_n_bins_times_samples_per_bin():
    rng = np.random.default_rng(0)
    n = 5000
    inputs = rng.random((n, 3))
    targets = rng.random((n, 1))
    errors = rng.random(n)  # plenty of distinct values -> every bin populated

    buffer = ReplayBuffer(n_bins=20, samples_per_bin=100, rng=np.random.default_rng(1))
    buffer.add_task(inputs, targets, errors)

    assert len(buffer) == 20 * 100


def test_add_task_caps_at_bin_size_when_a_bin_has_fewer_rows_than_requested():
    # Two distinct error values -> at most 2 non-empty bins regardless of
    # n_bins, and each bin can only contribute as many rows as it has.
    n = 30
    inputs = np.zeros((n, 1))
    targets = np.zeros((n, 1))
    errors = np.concatenate([np.zeros(10), np.ones(20)])

    buffer = ReplayBuffer(n_bins=20, samples_per_bin=100, rng=np.random.default_rng(0))
    buffer.add_task(inputs, targets, errors)

    assert len(buffer) == 30  # can't exceed what's actually available


def test_add_task_selects_without_replacement_within_a_bin():
    n = 50
    inputs = np.arange(n).reshape(-1, 1).astype(float)  # unique per-row id
    targets = np.zeros((n, 1))
    errors = np.zeros(n)  # single bin, all rows tied

    buffer = ReplayBuffer(n_bins=1, samples_per_bin=10, rng=np.random.default_rng(0))
    buffer.add_task(inputs, targets, errors)

    replay_inputs, _, _ = buffer.get_replay_data()
    assert len(replay_inputs) == 10
    assert len(np.unique(replay_inputs)) == 10  # no duplicate rows picked


def test_buffer_grows_across_tasks_with_correct_task_ids():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(n_bins=4, samples_per_bin=5, rng=rng)

    for task in range(3):
        n = 200
        inputs = rng.random((n, 2))
        targets = rng.random((n, 1))
        errors = rng.random(n)
        buffer.add_task(inputs, targets, errors)

    assert len(buffer) == 3 * 4 * 5

    replay_inputs, replay_targets, task_ids = buffer.get_replay_data()
    assert len(replay_inputs) == len(replay_targets) == len(task_ids) == len(buffer)
    assert set(np.unique(task_ids).tolist()) == {0, 1, 2}
    # growing: task 0's 20 rows always precede task 1's, which precede task 2's
    assert list(task_ids) == sorted(task_ids.tolist())


def test_get_replay_data_returns_none_when_empty():
    buffer = ReplayBuffer()
    inputs, targets, task_ids = buffer.get_replay_data()
    assert inputs is None
    assert targets is None
    assert task_ids is None
    assert len(buffer) == 0


def test_add_task_respects_explicit_task_id():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(n_bins=2, samples_per_bin=3, rng=rng)
    inputs = rng.random((50, 1))
    targets = rng.random((50, 1))
    errors = rng.random(50)

    buffer.add_task(inputs, targets, errors, task_id=7)

    _, _, task_ids = buffer.get_replay_data()
    assert set(np.unique(task_ids).tolist()) == {7}


def test_add_task_with_regime_boundaries_still_totals_n_bins_times_samples_per_bin():
    # Same total budget whether the task is one regime or several -- just
    # redistributed, never n_bins*samples_per_bin PER regime.
    rng = np.random.default_rng(0)
    n = 5000
    inputs = rng.random((n, 3))
    targets = rng.random((n, 1))
    errors = rng.random(n)

    buffer = ReplayBuffer(n_bins=20, samples_per_bin=100, rng=np.random.default_rng(1))
    buffer.add_task(inputs, targets, errors, regime_boundaries=[1000, 3000])

    assert len(buffer) == pytest.approx(20 * 100, abs=3)  # rounding across 3 regimes


def test_add_task_splits_quota_across_regimes_proportional_to_size():
    rng = np.random.default_rng(0)
    n = 4000
    inputs = rng.random((n, 2))
    targets = rng.random((n, 1))
    errors = rng.random(n)
    # Regime 1: rows [0, 800) -- 20% of the task. Regime 2: [800, 4000) -- 80%.
    boundary = 800

    buffer = ReplayBuffer(n_bins=20, samples_per_bin=100, rng=np.random.default_rng(2))
    buffer.add_task(inputs, targets, errors, regime_boundaries=[boundary])

    replay_inputs, _, _ = buffer.get_replay_data()
    # Every selected row's input matches a row from either regime 1 or
    # regime 2 -- count how many came from each by matching back to the
    # original array (inputs are unique per row via rng.random).
    from_regime_1 = sum(np.any(np.all(inputs[:boundary] == row, axis=1)) for row in replay_inputs)
    total = len(replay_inputs)
    # ~20% of ~2000 total should come from regime 1 (proportional to its 800/4000 share).
    assert 300 <= from_regime_1 <= 500
    assert 1500 <= total <= 2000


def test_add_task_with_regime_boundaries_draws_from_every_regime():
    # A regime that's small but non-empty must still contribute SOME rows,
    # not get starved to zero by rounding.
    n = 100
    inputs = np.arange(n).reshape(-1, 1).astype(float)
    targets = np.zeros((n, 1))
    errors = np.arange(n).astype(float)  # monotonic -> clean quantile bins
    # Regime 1: rows [0, 90). Regime 2: rows [90, 100) -- small but real.
    boundary = 90

    buffer = ReplayBuffer(n_bins=5, samples_per_bin=4, rng=np.random.default_rng(0))
    buffer.add_task(inputs, targets, errors, regime_boundaries=[boundary])

    replay_inputs, _, _ = buffer.get_replay_data()
    assert np.any(replay_inputs[:, 0] < boundary)
    assert np.any(replay_inputs[:, 0] >= boundary)


def test_add_task_without_regime_boundaries_is_unchanged_single_regime_behavior():
    # regime_boundaries=None (the default/common case) must select EXACTLY
    # the same rows as the pre-regime-aware behavior -- no regression for
    # single-regime tasks.
    rng = np.random.default_rng(0)
    n = 1000
    inputs = rng.random((n, 2))
    targets = rng.random((n, 1))
    errors = rng.random(n)

    buffer_a = ReplayBuffer(n_bins=10, samples_per_bin=5, rng=np.random.default_rng(42))
    buffer_a.add_task(inputs, targets, errors)

    buffer_b = ReplayBuffer(n_bins=10, samples_per_bin=5, rng=np.random.default_rng(42))
    buffer_b.add_task(inputs, targets, errors, regime_boundaries=[])

    inputs_a, _, _ = buffer_a.get_replay_data()
    inputs_b, _, _ = buffer_b.get_replay_data()
    np.testing.assert_array_equal(inputs_a, inputs_b)


def test_add_task_with_context_cols_splits_quota_by_context_value_quantile():
    # A context column whose value is independent of (and interleaved
    # with) row order -- context-based groups are generally
    # NON-contiguous, unlike regime_boundaries' contiguous slices.
    rng = np.random.default_rng(0)
    n = 4000
    context = rng.random(n)  # continuous PV value, e.g. SIOC:SYS0:ML00:AO627
    inputs = np.column_stack([rng.random(n), rng.random(n), context])
    targets = rng.random((n, 1))
    errors = rng.random(n)

    buffer = ReplayBuffer(n_bins=20, samples_per_bin=100, rng=np.random.default_rng(1))
    buffer.add_task(inputs, targets, errors, context_cols=[2], context_bins=4)

    replay_inputs, _, _ = buffer.get_replay_data()
    assert len(replay_inputs) == pytest.approx(20 * 100, abs=5)

    # Every context-value quartile is represented, roughly evenly (quantile
    # binning makes each quartile ~25% of this task's rows).
    ctx_vals = replay_inputs[:, 2]
    edges = np.quantile(context, [0.0, 0.25, 0.5, 0.75, 1.0])
    counts = []
    for j in range(4):
        mask = (
            (ctx_vals >= edges[0]) & (ctx_vals <= edges[1]) if j == 0
            else (ctx_vals > edges[j]) & (ctx_vals <= edges[j + 1])
        )
        counts.append(mask.sum())
    assert all(c > 0 for c in counts)
    assert max(counts) - min(counts) <= 0.1 * len(replay_inputs)


def test_add_task_with_multiple_context_cols_groups_by_joint_combination():
    rng = np.random.default_rng(2)
    n = 2000
    ctx_a = rng.random(n)
    ctx_b = rng.random(n)
    inputs = np.column_stack([rng.random(n), ctx_a, ctx_b])
    targets = rng.random((n, 1))
    errors = rng.random(n)

    buffer = ReplayBuffer(n_bins=8, samples_per_bin=10, rng=np.random.default_rng(3))
    buffer.add_task(inputs, targets, errors, context_cols=[1, 2], context_bins=2)

    replay_inputs, _, _ = buffer.get_replay_data()
    # 2 context vars x 2 bins each -> up to 4 joint groups; every
    # combination that actually has rows should contribute something.
    a_edges = np.quantile(ctx_a, [0.0, 0.5, 1.0])
    b_edges = np.quantile(ctx_b, [0.0, 0.5, 1.0])
    a_bin = (replay_inputs[:, 1] > a_edges[1]).astype(int)
    b_bin = (replay_inputs[:, 2] > b_edges[1]).astype(int)
    assert set(zip(a_bin.tolist(), b_bin.tolist())) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_add_task_raises_when_both_regime_boundaries_and_context_cols_given():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(n_bins=2, samples_per_bin=3, rng=rng)
    inputs = rng.random((20, 2))
    targets = np.zeros((20, 1))
    errors = np.arange(20.0)

    with pytest.raises(ValueError):
        buffer.add_task(inputs, targets, errors, regime_boundaries=[10], context_cols=[0])


def test_add_task_with_out_of_range_regime_boundary_raises_rather_than_silently_clipping():
    # Pre-existing, intentionally-unchanged subtlety of the refactor from
    # contiguous (start, end) slices to np.arange(start, end) groups:
    # scores[start:end] used to silently clip an out-of-range boundary;
    # np.arange(start, end) does not. Only reachable with a malformed
    # regime_boundaries (a value > len(scores)) -- real callers
    # (RegimeAwareTrainValSplitPreprocessor) never produce one.
    rng = np.random.default_rng(0)
    n = 100
    inputs = rng.random((n, 2))
    targets = np.zeros((n, 1))
    errors = rng.random(n)

    buffer = ReplayBuffer(n_bins=2, samples_per_bin=3, rng=rng)
    with pytest.raises(IndexError):
        buffer.add_task(inputs, targets, errors, regime_boundaries=[50, 150])
