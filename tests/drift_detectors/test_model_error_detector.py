import numpy as np
import pytest

from l3kit.drift_detectors.model_error_detector import ModelErrorDriftDetector


class _ZeroPredModel:
    """run() always predicts 0 -- makes per-sample absolute error exactly
    equal to |target|, easy to reason about in tests."""

    def run(self, inputs):
        n = len(inputs)
        return np.zeros(n), np.zeros(n)


def test_fit_computes_error_via_model_run_and_sets_125_percent_threshold():
    train_input = np.zeros((200, 1))
    train_target = np.full(200, 0.4)
    train_target[100] = 2.0  # one elevated error -- max() should latch onto it

    detector = ModelErrorDriftDetector(rolling_window=5)
    threshold = detector.fit(model=_ZeroPredModel(), train_inputs=train_input, train_targets=train_target)

    assert threshold == pytest.approx(1.25 * 0.72)
    assert detector.threshold == threshold


def test_fit_raises_without_all_three_required_args():
    detector = ModelErrorDriftDetector()
    with pytest.raises(ValueError):
        detector.fit(model=_ZeroPredModel(), train_inputs=np.zeros((10, 1)))


def test_run_raises_without_targets():
    detector = ModelErrorDriftDetector(rolling_window=5)
    detector.threshold = 1.0
    with pytest.raises(ValueError):
        detector.run(model=_ZeroPredModel(), inputs=np.zeros((10, 1)))


def test_run_uses_precomputed_preds_without_calling_model_run():
    class _ExplodingModel:
        def run(self, inputs):
            raise AssertionError("should not be called when preds is precomputed")

    detector = ModelErrorDriftDetector(rolling_window=5)
    detector.threshold = 1.0

    idx = detector.run(
        model=_ExplodingModel(), inputs=np.zeros((20, 1)), targets=np.full(20, 5.0), preds=np.zeros(20),
    )

    assert idx is not None


def test_run_computes_error_via_model_when_preds_not_precomputed():
    detector = ModelErrorDriftDetector(rolling_window=5)
    detector.threshold = 1.0
    targets = np.concatenate([np.full(60, 0.1), np.full(40, 5.0)])

    idx = detector.run(model=_ZeroPredModel(), inputs=np.zeros((100, 1)), targets=targets)

    assert idx is not None
    assert idx >= 60


def test_error_smooths_target_and_pred_before_diffing_not_after():
    # Alternating target/pred pairs whose raw per-sample errors are all
    # identical (rolling_mean(|target - pred|) would be flat), but whose
    # smoothed-then-diffed values (RegressionEvaluator's mae_curve
    # convention) are not -- distinguishes the two orderings.
    rng = np.random.default_rng(0)
    n = 50
    target = rng.normal(size=n)
    pred = target + np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # constant |error| = 1.0 everywhere

    detector = ModelErrorDriftDetector(rolling_window=10)

    class _FixedPredModel:
        def run(self, inputs):
            return pred, np.zeros(n)

    threshold = detector.fit(model=_FixedPredModel(), train_inputs=np.zeros((n, 1)), train_targets=target)

    # rolling_mean(|target - pred|) would be a flat 1.0 everywhere ->
    # threshold == 1.25. Smoothing target/pred first (their alternating +-1
    # offsets partially cancel in the rolling mean) gives a threshold well
    # below that.
    assert threshold < 1.25


def test_sustained_for_defaults_to_one_sample():
    # No persistence requirement unless explicitly asked for -- a bare
    # ModelErrorDriftDetector() behaves like before this feature.
    detector = ModelErrorDriftDetector(rolling_window=5)
    detector.threshold = 1.0

    idx = detector.run(model=_ZeroPredModel(), inputs=np.zeros((20, 1)), targets=np.full(20, 5.0))

    assert idx is not None


def test_sustained_for_is_forwarded_to_the_internal_threshold_detector():
    detector = ModelErrorDriftDetector(rolling_window=1, sustained_for=10)
    detector.threshold = 1.0
    # Only 5 consecutive elevated targets -- short of sustained_for=10.
    targets = np.concatenate([np.full(20, 0.1), np.full(5, 5.0), np.full(20, 0.1)])

    idx = detector.run(model=_ZeroPredModel(), inputs=np.zeros((45, 1)), targets=targets)

    assert idx is None
