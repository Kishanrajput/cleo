import numpy as np

from cleo.signal import rolling_mean

from .base import BaseDriftDetector
from .threshold_detector import ThresholdDriftDetector


def _smoothed_absolute_error(targets, preds, window_size):
    """|rolling_mean(target) - rolling_mean(pred)| -- smooth target and pred
    FIRST, then diff, matching RegressionEvaluator's own mae_curve
    convention (a regression evaluator). NOT the same as
    rolling_mean(|target - pred|) (smoothing the raw per-sample error after
    the fact), which is what this used to compute."""
    smoothed_target = rolling_mean(np.asarray(targets).flatten(), window_size=window_size)
    smoothed_preds = rolling_mean(np.asarray(preds).flatten(), window_size=window_size)
    return np.abs(smoothed_target - smoothed_preds)


class ModelErrorDriftDetector(BaseDriftDetector):
    """Drift = |rolling_mean(target) - rolling_mean(pred)| (same smoothing
    convention as RegressionEvaluator's mae_curve, see _smoothed_absolute_error
    above) crossing 125% of its own max rolling-mean value over the base
    model's training data, sustained for `sustained_for` consecutive samples
    (default 1 -- a single crossing is enough). Same mechanics as
    ModelUncertaintyDriftDetector (shares its ThresholdDriftDetector engine),
    just fed error instead of uncertainty -- the error signal handed to it is
    already smoothed, so its internal ThresholdDriftDetector is given
    rolling_window=1 (a no-op) and only reused for the
    multiplier-threshold/sustained-crossing bookkeeping.

    Needs ground truth targets both to fit (training data) and to run --
    whatever's being scanned must have a recorded true value to compare
    against (e.g. historical logged data, which is what this project scans),
    not usable on genuinely unlabeled live data."""

    def __init__(self, rolling_window=50, multiplier=1.25, sustained_for=1, hard_threshold=None, **config):
        super().__init__(
            rolling_window=rolling_window, multiplier=multiplier, sustained_for=sustained_for,
            hard_threshold=hard_threshold, **config,
        )
        self.rolling_window = rolling_window
        self._threshold_detector = ThresholdDriftDetector(
            rolling_window=1, multiplier=multiplier, hard_threshold=hard_threshold, sustained_for=sustained_for,
        )

    @property
    def threshold(self):
        return self._threshold_detector.threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold_detector.threshold = value

    def fit(self, model=None, train_inputs=None, train_targets=None, **kwargs):
        if model is None or train_inputs is None or train_targets is None:
            raise ValueError("ModelErrorDriftDetector.fit() requires model, train_inputs, and train_targets.")
        preds, _ = model.run(train_inputs)
        return self._threshold_detector.fit(_smoothed_absolute_error(train_targets, preds, self.rolling_window))

    def run(self, offset=0, model=None, inputs=None, targets=None, std=None, preds=None, **kwargs):
        if targets is None:
            raise ValueError("ModelErrorDriftDetector.run() requires targets (ground truth) to compute error.")
        if preds is None:
            if model is None or inputs is None:
                raise ValueError("ModelErrorDriftDetector.run() requires either `preds` or (`model`, `inputs`).")
            preds, _ = model.run(inputs)
        return self._threshold_detector.run(_smoothed_absolute_error(targets, preds, self.rolling_window), offset=offset)
