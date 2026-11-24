import numpy as np
import pytest

from l3kit.drift_detectors.threshold_detector import ThresholdDriftDetector


def test_fit_is_125_percent_of_max_rolling_mean_over_the_reference_signal():
    train_input = np.full(200, 0.4)
    train_input[100] = 2.0  # one elevated row -- max() should latch onto it

    detector = ThresholdDriftDetector(rolling_window=5, multiplier=1.25)
    threshold = detector.fit(train_input)

    # rolling_mean(window=5) peaks when the elevated row is inside the
    # trailing window: (4*0.4 + 2.0) / 5 = 0.72
    assert threshold == pytest.approx(1.25 * 0.72)
    assert detector.threshold == threshold


def test_fit_uses_the_entire_reference_signal():
    # The large early values must affect the result -- the whole reference
    # signal feeds fit(), not just a recent slice.
    old_large = np.full(900, 5.0)
    recent_small = np.full(100, 0.1)
    reference_signal = np.concatenate([old_large, recent_small])

    detector = ThresholdDriftDetector(rolling_window=5)
    threshold = detector.fit(reference_signal)

    assert threshold == pytest.approx(1.25 * 5.0)


def test_run_raises_if_fit_was_never_called():
    detector = ThresholdDriftDetector(rolling_window=5)
    try:
        detector.run(np.full(20, 0.1))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_run_returns_none_when_signal_never_crosses_threshold():
    detector = ThresholdDriftDetector(rolling_window=10)
    detector.threshold = 1.0

    idx = detector.run(np.full(100, 0.1))

    assert idx is None


def test_run_finds_known_crossing():
    detector = ThresholdDriftDetector(rolling_window=10)
    detector.threshold = 1.0
    signal = np.concatenate([np.full(60, 0.1), np.full(40, 5.0)])

    idx = detector.run(signal)

    assert idx is not None
    # rolling_mean(window=10) only exceeds 1.0 once enough high values have
    # entered the window -- first crossing should land at/after index 60.
    assert idx >= 60


def test_run_respects_offset_for_incremental_scanning():
    detector = ThresholdDriftDetector(rolling_window=10)
    detector.threshold = 1.0
    # A brief spike that returns to normal -- simulates "one file's worth of
    # elevated signal, then back to baseline in the next file".
    signal = np.concatenate([np.full(60, 0.1), np.full(5, 5.0), np.full(60, 0.1)])

    full_idx = detector.run(signal, offset=0)
    assert full_idx is not None

    # offset before the crossing still finds the exact same index.
    assert detector.run(signal, offset=30) == full_idx

    # offset placed after the rolling mean has decayed back below threshold
    # finds nothing new -- matches "already scanned this region, and it's
    # calmed back down" rather than re-flagging it.
    decayed_offset = full_idx + 15
    assert detector.run(signal, offset=decayed_offset) is None


def test_sustained_for_ignores_a_brief_spike_that_decays_back_down():
    detector = ThresholdDriftDetector(rolling_window=1, sustained_for=30)
    detector.threshold = 1.0
    # A brief spike (10 samples, well under sustained_for) that returns to
    # normal -- a single-sample-crossing detector (sustained_for=1) would
    # fire on this; a sustained one should not.
    signal = np.concatenate([np.full(60, 0.1), np.full(10, 5.0), np.full(60, 0.1)])

    assert detector.run(signal) is None


def test_sustained_for_confirms_a_persistent_shift_and_dates_it_from_the_start():
    detector = ThresholdDriftDetector(rolling_window=1, sustained_for=30)
    detector.threshold = 1.0
    signal = np.concatenate([np.full(60, 0.1), np.full(40, 5.0)])  # 40 >= sustained_for, never decays

    idx = detector.run(signal)

    # Dated from where the sustained run BEGINS (index 60), not from index
    # 60 + sustained_for - 1 where it's actually confirmed.
    assert idx == 60


def test_sustained_for_one_matches_original_single_sample_behavior():
    signal = np.concatenate([np.full(60, 0.1), np.full(5, 5.0), np.full(60, 0.1)])

    default_detector = ThresholdDriftDetector(rolling_window=1)
    default_detector.threshold = 1.0
    explicit_detector = ThresholdDriftDetector(rolling_window=1, sustained_for=1)
    explicit_detector.threshold = 1.0

    assert default_detector.run(signal) == 60
    assert explicit_detector.run(signal) == 60
