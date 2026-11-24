from fel_cl.drift_detectors import CombinedDriftDetector, DebouncedDriftDetector, build_combined_detector, build_single_detector


def test_single_detector_debounced_only_for_ood_ids():
    assert isinstance(build_single_detector("input_ood", rolling_window=5), DebouncedDriftDetector)
    assert not isinstance(build_single_detector("model_error", rolling_window=5), DebouncedDriftDetector)


def test_combined_detector_has_one_entry_per_id():
    det = build_combined_detector(["model_error", "output_ood"], rolling_window=5)
    assert isinstance(det, CombinedDriftDetector) and set(det.detectors) == {"model_error", "output_ood"}
