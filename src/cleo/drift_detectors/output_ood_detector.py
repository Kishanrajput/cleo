import numpy as np

from cleo.novelty import fit_novelty_reference, novelty_score

from .base import BaseDriftDetector
from .adwin_detector import AdwinDriftDetector


def _as_columns(values):
    values = np.asarray(values)
    return values.reshape(len(values), -1)


class OutputOODDriftDetector(BaseDriftDetector):
    """Same idea as InputOODDriftDetector, on the raw measured output
    (target) values instead of the input features -- flags when the
    observed output itself has moved away from its training distribution,
    independent of model performance (unlike ModelErrorDriftDetector, which
    compares the output against the model's prediction rather than against
    its own history). Same smoothing + small-delta false-positive-reduction
    recipe; wrap in DebouncedDriftDetector for an additional layer."""

    def __init__(self, delta=1e-20, rolling_window=50, **config):
        super().__init__(delta=delta, rolling_window=rolling_window, **config)
        self._adwin_detector = AdwinDriftDetector(delta=delta, rolling_window=rolling_window)
        self._ref_mean = None
        self._ref_std = None

    def fit(self, model=None, train_inputs=None, train_targets=None, **kwargs):
        if train_targets is None:
            raise ValueError("OutputOODDriftDetector.fit() requires train_targets.")
        self._ref_mean, self._ref_std = fit_novelty_reference(_as_columns(train_targets))

    def run(self, offset=0, model=None, inputs=None, targets=None, std=None, preds=None, **kwargs):
        if self._ref_mean is None:
            raise RuntimeError("OutputOODDriftDetector.fit(train_targets=...) must be called before run().")
        if targets is None:
            raise ValueError("OutputOODDriftDetector.run() requires targets.")
        score = novelty_score(_as_columns(targets), self._ref_mean, self._ref_std)
        return self._adwin_detector.run(score, offset=offset)
