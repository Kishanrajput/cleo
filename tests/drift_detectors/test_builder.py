from l3kit.drift_detectors import CombinedDriftDetector, DebouncedDriftDetector, build_combined_detector, build_single_detector


def test_single_detector_debounced_only_for_ood_ids():
    assert isinstance(build_single_detector("input_ood", rolling_window=5), DebouncedDriftDetector)
    assert not isinstance(build_single_detector("model_error", rolling_window=5), DebouncedDriftDetector)


def test_combined_detector_has_one_entry_per_id():
    det = build_combined_detector(["model_error", "output_ood"], rolling_window=5)
    assert isinstance(det, CombinedDriftDetector) and set(det.detectors) == {"model_error", "output_ood"}


def test_hard_threshold_caps_fitted_threshold():
    import numpy as np

    class _Model:
        def run(self, x):
            return np.zeros(len(x)), np.full(len(x), 0.9)

    capped = build_single_detector("model_uncertainty", rolling_window=2, hard_threshold=0.2)
    free = build_single_detector("model_uncertainty", rolling_window=2)
    for det in (capped, free):
        det.fit(model=_Model(), train_inputs=np.zeros((10, 1)))
    assert capped.threshold == 0.2
    assert free.threshold > 0.2
