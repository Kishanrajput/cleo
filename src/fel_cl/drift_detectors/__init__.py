from .base import BaseDriftDetector
from fel_cl.registry import register, make, spec, list_registered_modules

# Low-level primitives -- raw-signal-in, index-out. Not meant to be composed
# directly into a CombinedDriftDetector (see their docstrings); the named
# detectors below wrap them to compute the right signal from model/inputs/targets.
register(id="threshold", entry_point="fel_cl.drift_detectors.threshold_detector:ThresholdDriftDetector")
register(id="adwin", entry_point="fel_cl.drift_detectors.adwin_detector:AdwinDriftDetector")

# Named drift-detection strategies -- combine any subset via "combined".
register(
    id="model_uncertainty",
    entry_point="fel_cl.drift_detectors.model_uncertainty_detector:ModelUncertaintyDriftDetector",
)
register(id="model_error", entry_point="fel_cl.drift_detectors.model_error_detector:ModelErrorDriftDetector")
register(id="input_ood", entry_point="fel_cl.drift_detectors.input_ood_detector:InputOODDriftDetector")
register(id="output_ood", entry_point="fel_cl.drift_detectors.output_ood_detector:OutputOODDriftDetector")

# Wrappers/combinators.
register(id="debounced", entry_point="fel_cl.drift_detectors.debounced_detector:DebouncedDriftDetector")
register(id="combined", entry_point="fel_cl.drift_detectors.combined_detector:CombinedDriftDetector")

__all__ = ["BaseDriftDetector", "register", "make", "spec", "list_registered_modules"]
from .builder import build_single_detector, build_combined_detector, DEBOUNCED_DETECTOR_IDS  # noqa: E402
from .combined_detector import CombinedDriftDetector  # noqa: E402
from .debounced_detector import DebouncedDriftDetector  # noqa: E402

__all__ += ["build_single_detector", "build_combined_detector", "DEBOUNCED_DETECTOR_IDS", "CombinedDriftDetector", "DebouncedDriftDetector"]
