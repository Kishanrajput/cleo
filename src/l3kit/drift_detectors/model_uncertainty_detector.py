from .base import BaseDriftDetector
from .threshold_detector import ThresholdDriftDetector


class ModelUncertaintyDriftDetector(BaseDriftDetector):
    """Drift = the model's predicted uncertainty (std) crossing 125% of its
    own max rolling-mean value over the base model's training data (the
    original notebook logic), sustained for `sustained_for` consecutive
    samples (default 1 -- a single crossing is enough). A thin, model-aware
    wrapper around ThresholdDriftDetector: computes its own signal (std, via
    model.run()) instead of requiring the caller to precompute and pass one."""

    def __init__(self, rolling_window=50, multiplier=1.25, sustained_for=1, hard_threshold=None, **config):
        super().__init__(
            rolling_window=rolling_window, multiplier=multiplier, sustained_for=sustained_for,
            hard_threshold=hard_threshold, **config,
        )
        self._threshold_detector = ThresholdDriftDetector(
            rolling_window=rolling_window, multiplier=multiplier, hard_threshold=hard_threshold, sustained_for=sustained_for,
        )

    @property
    def threshold(self):
        return self._threshold_detector.threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold_detector.threshold = value

    def fit(self, model=None, train_inputs=None, train_targets=None, **kwargs):
        if model is None or train_inputs is None:
            raise ValueError("ModelUncertaintyDriftDetector.fit() requires model and train_inputs.")
        _, std = model.run(train_inputs)
        return self._threshold_detector.fit(std)

    def run(self, offset=0, model=None, inputs=None, targets=None, std=None, preds=None, **kwargs):
        if std is None:
            if model is None or inputs is None:
                raise ValueError("ModelUncertaintyDriftDetector.run() requires either `std` or (`model`, `inputs`).")
            _, std = model.run(inputs)
        return self._threshold_detector.run(std, offset=offset)
