from . import make

# ADWIN-based OOD detectors fire more readily on one noisy chunk than the
# threshold-based ones (which already need a sustained crossing), so they are
# wrapped in DebouncedDriftDetector.
DEBOUNCED_DETECTOR_IDS = {"input_ood", "output_ood"}


def build_single_detector(
    detector_id, rolling_window, multiplier=1.25, delta=1e-20, sustained_for=30,
    debounce_confirm_for=2, debounce_lookback=6, hard_threshold=None,
):
    """One registered detector, debounced if its id is in DEBOUNCED_DETECTOR_IDS.
    All detector-family kwargs are passed uniformly; each concrete class picks
    out only what it declares (e.g. sustained_for is ignored by ADWIN-based ones).
    hard_threshold: optional ceiling on the fitted threshold (model_error / model_uncertainty only)."""
    detector = make(
        detector_id, rolling_window=rolling_window, multiplier=multiplier, delta=delta, sustained_for=sustained_for,
        hard_threshold=hard_threshold,
    )
    if detector_id in DEBOUNCED_DETECTOR_IDS:
        detector = make("debounced", detector=detector, confirm_for=debounce_confirm_for, lookback=debounce_lookback)
    return detector


def build_combined_detector(detector_ids, hard_thresholds=None, **kwargs):
    """OR-combination of one build_single_detector per id. hard_thresholds:
    optional {detector_id: ceiling} applied per detector."""
    hard_thresholds = hard_thresholds or {}
    return make("combined", detectors={
        i: build_single_detector(i, hard_threshold=hard_thresholds.get(i), **kwargs) for i in detector_ids
    })
