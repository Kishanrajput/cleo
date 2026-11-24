import numpy as np
from river.drift import ADWIN

from fel_cl.signal import rolling_mean

from .base import BaseDriftDetector


class AdwinDriftDetector(BaseDriftDetector):
    """ADWIN (river.drift.ADWIN) fed the rolling-mean of the signal. Stateful
    and incremental: each instance owns one ADWIN detector that's fed
    forward from wherever the previous run() call left off (via `offset`),
    so repeated calls against a growing signal don't re-feed already-seen
    values. Construct a fresh instance per scan cycle.

    Low-level primitive: operates on a raw numeric signal array the caller
    already computed, not on a model/inputs/targets. InputOODDriftDetector
    and OutputOODDriftDetector wrap an instance of this to do the novelty-
    score computation for you -- use those (registry ids "input_ood" /
    "output_ood") rather than this directly in a CombinedDriftDetector,
    whose kwargs-forwarding convention this class's run() signature doesn't
    match."""

    def __init__(self, delta=1e-20, rolling_window=50, **config):
        super().__init__(delta=delta, rolling_window=rolling_window, **config)
        self.delta = delta
        self.rolling_window = rolling_window
        self._adwin = ADWIN(delta=delta)

    def run(self, signal, offset=0):
        rolling = rolling_mean(signal, window_size=self.rolling_window)
        for idx in range(offset, len(rolling)):
            value = rolling[idx]
            if np.isnan(value):
                continue
            self._adwin.update(value)
            if self._adwin.drift_detected:
                return idx
        return None
