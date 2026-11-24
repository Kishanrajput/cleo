import numpy as np
import pytest

from fel_cl.drift_detectors.output_ood_detector import OutputOODDriftDetector


def test_fit_requires_train_targets():
    detector = OutputOODDriftDetector()
    with pytest.raises(ValueError):
        detector.fit()


def test_run_raises_if_fit_was_never_called():
    detector = OutputOODDriftDetector()
    with pytest.raises(RuntimeError):
        detector.run(targets=np.zeros(10))


def test_run_raises_without_targets():
    detector = OutputOODDriftDetector()
    detector.fit(train_targets=np.random.default_rng(0).normal(size=200))
    with pytest.raises(ValueError):
        detector.run()


def test_run_fires_on_a_sustained_output_distribution_shift():
    rng = np.random.default_rng(0)
    train_targets = rng.normal(loc=0.0, scale=1.0, size=300)
    detector = OutputOODDriftDetector(delta=0.01, rolling_window=1)
    detector.fit(train_targets=train_targets)

    shifted = np.concatenate([
        rng.normal(loc=0.0, scale=1.0, size=150),
        rng.normal(loc=20.0, scale=1.0, size=150),
    ])

    idx = detector.run(targets=shifted)

    assert idx is not None
    assert idx >= 150


def test_run_does_not_fire_when_output_distribution_is_unchanged():
    rng = np.random.default_rng(1)
    train_targets = rng.normal(loc=0.0, scale=1.0, size=300)
    detector = OutputOODDriftDetector(delta=0.01, rolling_window=1)
    detector.fit(train_targets=train_targets)

    unchanged = rng.normal(loc=0.0, scale=1.0, size=300)
    idx = detector.run(targets=unchanged)

    assert idx is None


def test_handles_single_column_target_shape():
    # targets often arrive as (n, 1) from to_numpy preprocessing, not (n,).
    rng = np.random.default_rng(2)
    train_targets = rng.normal(size=(200, 1))
    detector = OutputOODDriftDetector(delta=0.01, rolling_window=1)
    detector.fit(train_targets=train_targets)

    idx = detector.run(targets=rng.normal(size=(50, 1)))

    assert idx is None or isinstance(idx, int)
