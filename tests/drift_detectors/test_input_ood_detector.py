import numpy as np
import pytest

from cleo.drift_detectors.input_ood_detector import InputOODDriftDetector


def test_fit_requires_train_inputs():
    detector = InputOODDriftDetector()
    with pytest.raises(ValueError):
        detector.fit()


def test_run_raises_if_fit_was_never_called():
    detector = InputOODDriftDetector()
    with pytest.raises(RuntimeError):
        detector.run(inputs=np.zeros((10, 3)))


def test_run_raises_without_inputs():
    detector = InputOODDriftDetector()
    detector.fit(train_inputs=np.random.default_rng(0).normal(size=(200, 3)))
    with pytest.raises(ValueError):
        detector.run()


def test_run_fires_on_a_sustained_input_distribution_shift():
    rng = np.random.default_rng(0)
    train_inputs = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    detector = InputOODDriftDetector(delta=0.01, rolling_window=1)
    detector.fit(train_inputs=train_inputs)

    shifted = np.concatenate([
        rng.normal(loc=0.0, scale=1.0, size=(150, 3)),
        rng.normal(loc=20.0, scale=1.0, size=(150, 3)),
    ])

    idx = detector.run(inputs=shifted)

    assert idx is not None
    assert idx >= 150


def test_run_does_not_fire_when_input_distribution_is_unchanged():
    rng = np.random.default_rng(1)
    train_inputs = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    detector = InputOODDriftDetector(delta=0.01, rolling_window=1)
    detector.fit(train_inputs=train_inputs)

    unchanged = rng.normal(loc=0.0, scale=1.0, size=(300, 3))
    idx = detector.run(inputs=unchanged)

    assert idx is None
