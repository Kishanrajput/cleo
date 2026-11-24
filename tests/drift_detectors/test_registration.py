from cleo.drift_detectors import list_registered_modules, spec


def test_threshold_detector_registered():
    assert "threshold" in list_registered_modules()
    assert spec("threshold").entry_point == "cleo.drift_detectors.threshold_detector:ThresholdDriftDetector"


def test_adwin_detector_registered():
    assert "adwin" in list_registered_modules()
    assert spec("adwin").entry_point == "cleo.drift_detectors.adwin_detector:AdwinDriftDetector"


def test_combined_detector_registered():
    assert "combined" in list_registered_modules()
    assert spec("combined").entry_point == "cleo.drift_detectors.combined_detector:CombinedDriftDetector"


def test_model_uncertainty_detector_registered():
    assert "model_uncertainty" in list_registered_modules()
    assert spec("model_uncertainty").entry_point == (
        "cleo.drift_detectors.model_uncertainty_detector:ModelUncertaintyDriftDetector"
    )


def test_model_error_detector_registered():
    assert "model_error" in list_registered_modules()
    assert spec("model_error").entry_point == "cleo.drift_detectors.model_error_detector:ModelErrorDriftDetector"


def test_input_ood_detector_registered():
    assert "input_ood" in list_registered_modules()
    assert spec("input_ood").entry_point == "cleo.drift_detectors.input_ood_detector:InputOODDriftDetector"


def test_output_ood_detector_registered():
    assert "output_ood" in list_registered_modules()
    assert spec("output_ood").entry_point == "cleo.drift_detectors.output_ood_detector:OutputOODDriftDetector"


def test_debounced_detector_registered():
    assert "debounced" in list_registered_modules()
    assert spec("debounced").entry_point == "cleo.drift_detectors.debounced_detector:DebouncedDriftDetector"
