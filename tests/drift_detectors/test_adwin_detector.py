import numpy as np

from l3kit.drift_detectors.adwin_detector import AdwinDriftDetector


def test_run_returns_index_of_first_signal_on_a_sustained_shift():
    rng = np.random.default_rng(0)
    signal = np.concatenate([rng.normal(0.0, 0.05, 300), rng.normal(5.0, 0.05, 300)])
    detector = AdwinDriftDetector(delta=0.01, rolling_window=1)

    idx = detector.run(signal)

    assert idx is not None
    assert 300 <= idx < 600  # fires sometime after the real shift, not before it


def test_run_returns_none_when_signal_never_shifts():
    signal = np.full(200, 0.1)
    detector = AdwinDriftDetector(delta=0.01, rolling_window=1)

    idx = detector.run(signal)

    assert idx is None


def test_run_respects_offset_for_incremental_feeding():
    # A spike entirely before offset -- nothing should fire from this call.
    signal = np.concatenate([np.full(50, 100.0), np.full(50, 0.1)])
    detector = AdwinDriftDetector(delta=1e-20, rolling_window=1)

    idx = detector.run(signal, offset=50)

    assert idx is None


def test_instance_is_stateful_across_repeated_calls():
    # Feeding the same growing signal in two calls (like the scan loop does,
    # one more file each time) should behave the same as one call with the
    # full signal and no offset -- the detector must not "forget" what it
    # already saw between calls.
    rng = np.random.default_rng(1)
    signal = np.concatenate([rng.normal(0.0, 0.05, 300), rng.normal(5.0, 0.05, 300)])

    detector_incremental = AdwinDriftDetector(delta=0.01, rolling_window=1)
    first_half_result = detector_incremental.run(signal[:300], offset=0)
    assert first_half_result is None  # no shift within the first half alone
    second_call_result = detector_incremental.run(signal, offset=300)

    detector_single_shot = AdwinDriftDetector(delta=0.01, rolling_window=1)
    single_shot_result = detector_single_shot.run(signal, offset=0)

    assert second_call_result == single_shot_result
