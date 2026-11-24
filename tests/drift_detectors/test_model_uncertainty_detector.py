import numpy as np
import pytest

from fel_cl.drift_detectors.model_uncertainty_detector import ModelUncertaintyDriftDetector


class _StdByRowFakeModel:
    """run() returns std equal to each row's first feature value -- lets a
    test control exactly what std the model exposes per row."""

    def run(self, inputs):
        std = np.asarray(inputs)[:, 0]
        preds = np.zeros(len(inputs))
        return preds, std


def test_fit_computes_std_via_model_run_and_sets_125_percent_threshold():
    train_input = np.full((200, 1), 0.4)
    train_input[100, 0] = 2.0  # one elevated row -- max() should latch onto it

    detector = ModelUncertaintyDriftDetector(rolling_window=5)
    threshold = detector.fit(model=_StdByRowFakeModel(), train_inputs=train_input)

    # rolling_mean(window=5) peaks when the elevated row is inside the
    # trailing window: (4*0.4 + 2.0) / 5 = 0.72
    assert threshold == pytest.approx(1.25 * 0.72)
    assert detector.threshold == threshold


def test_fit_raises_without_model_or_train_inputs():
    detector = ModelUncertaintyDriftDetector()
    with pytest.raises(ValueError):
        detector.fit(train_inputs=np.zeros((10, 1)))
    with pytest.raises(ValueError):
        detector.fit(model=_StdByRowFakeModel())


def test_run_uses_precomputed_std_without_calling_model_run():
    class _ExplodingModel:
        def run(self, inputs):
            raise AssertionError("should not be called when std is precomputed")

    detector = ModelUncertaintyDriftDetector(rolling_window=5)
    detector.threshold = 1.0

    idx = detector.run(model=_ExplodingModel(), inputs=np.zeros((20, 1)), std=np.full(20, 5.0))

    assert idx is not None


def test_run_computes_std_via_model_when_not_precomputed():
    detector = ModelUncertaintyDriftDetector(rolling_window=5)
    detector.threshold = 1.0
    inputs = np.concatenate([np.full((60, 1), 0.1), np.full((40, 1), 5.0)])

    idx = detector.run(model=_StdByRowFakeModel(), inputs=inputs)

    assert idx is not None
    assert idx >= 60


def test_run_returns_none_when_never_crosses():
    detector = ModelUncertaintyDriftDetector(rolling_window=5)
    detector.threshold = 1000.0

    idx = detector.run(model=_StdByRowFakeModel(), inputs=np.full((50, 1), 0.1))

    assert idx is None


def test_sustained_for_defaults_to_one_sample():
    # No persistence requirement unless explicitly asked for -- a bare
    # ModelUncertaintyDriftDetector() behaves like before this feature.
    detector = ModelUncertaintyDriftDetector(rolling_window=1)
    detector.threshold = 1.0

    idx = detector.run(model=_StdByRowFakeModel(), inputs=np.full((20, 1), 5.0))

    assert idx == 0


def test_sustained_for_is_forwarded_to_the_internal_threshold_detector():
    detector = ModelUncertaintyDriftDetector(rolling_window=1, sustained_for=10)
    detector.threshold = 1.0
    # Only 5 consecutive elevated rows -- short of sustained_for=10.
    inputs = np.concatenate([np.full((20, 1), 0.1), np.full((5, 1), 5.0), np.full((20, 1), 0.1)])

    idx = detector.run(model=_StdByRowFakeModel(), inputs=inputs)

    assert idx is None

