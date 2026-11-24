from .base import BaseDriftDetector


class CombinedDriftDetector(BaseDriftDetector):
    """OR-combination of named sub-detectors: fires as soon as ANY of them
    does, returning the earliest triggered index. After run(), `triggered`
    records which sub-detector(s) fired on that call, keyed by name --
    callers that want to know which specific detector(s) caused a trigger
    read it from there.

    Every sub-detector's fit()/run() is called with the SAME keyword
    arguments (model=, train_inputs=, train_targets=, inputs=, targets=,
    std=, preds=, offset=...) -- each concrete detector picks out only what
    it needs and ignores the rest (via **kwargs), so heterogeneous
    detectors (one needing a model, one needing only raw inputs, one
    needing targets) compose uniformly without the caller needing to know
    which is which.

    `detectors` is a dict of already-constructed detector instances (e.g.
    {"model_uncertainty": ModelUncertaintyDriftDetector(...), "input_ood":
    DebouncedDriftDetector(InputOODDriftDetector(...))}), not registry ids
    -- composing them (and deciding which combination to enable) is the
    driver's job, driven by config."""

    def __init__(self, detectors, **config):
        super().__init__(detectors=detectors, **config)
        self.detectors = detectors
        self.triggered = {}

    def fit(self, **kwargs):
        for detector in self.detectors.values():
            fit_fn = getattr(detector, "fit", None)
            if fit_fn is not None:
                fit_fn(**kwargs)

    def run(self, offset=0, **kwargs):
        results = {name: detector.run(offset=offset, **kwargs) for name, detector in self.detectors.items()}
        self.triggered = {name: idx is not None for name, idx in results.items()}
        candidates = [idx for idx in results.values() if idx is not None]
        return min(candidates) if candidates else None
