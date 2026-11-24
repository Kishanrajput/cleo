from fel_cl.novelty import fit_novelty_reference, novelty_score

from .base import BaseDriftDetector
from .adwin_detector import AdwinDriftDetector


class InputOODDriftDetector(BaseDriftDetector):
    """ADWIN-based out-of-distribution detection on the raw input features:
    a scalar novelty score (standardized distance from the base model's
    training input mean/std) fed through ADWIN via a rolling-mean-smoothed
    stream -- the smoothing plus a small, configurable `delta` are the same
    false-positive-reduction recipe already used by the uncertainty-based
    ADWIN detection this project relies on. For an additional layer of
    protection against a one-off spike, wrap this in DebouncedDriftDetector
    (see debounced_detector.py) rather than tuning delta alone."""

    def __init__(self, delta=1e-20, rolling_window=50, **config):
        super().__init__(delta=delta, rolling_window=rolling_window, **config)
        self._adwin_detector = AdwinDriftDetector(delta=delta, rolling_window=rolling_window)
        self._ref_mean = None
        self._ref_std = None

    def fit(self, model=None, train_inputs=None, train_targets=None, **kwargs):
        if train_inputs is None:
            raise ValueError("InputOODDriftDetector.fit() requires train_inputs.")
        self._ref_mean, self._ref_std = fit_novelty_reference(train_inputs)

    def run(self, offset=0, model=None, inputs=None, targets=None, std=None, preds=None, **kwargs):
        if self._ref_mean is None:
            raise RuntimeError("InputOODDriftDetector.fit(train_inputs=...) must be called before run().")
        if inputs is None:
            raise ValueError("InputOODDriftDetector.run() requires inputs.")
        score = novelty_score(inputs, self._ref_mean, self._ref_std)
        return self._adwin_detector.run(score, offset=offset)
