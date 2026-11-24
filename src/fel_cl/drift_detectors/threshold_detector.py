import numpy as np

from fel_cl.signal import rolling_mean

from .base import BaseDriftDetector


def _first_crossing(signal, threshold, window_size, offset=0, sustained_for=1):
    """First index (>= offset) where rolling_mean(signal, window_size)
    exceeds threshold for `sustained_for` CONSECUTIVE samples, or None if
    there isn't one. Returns the index where the sustained run BEGINS, not
    where it's confirmed (`sustained_for` samples later) -- consistent with
    DebouncedDriftDetector's own "date it from the first hit" convention.
    `offset` lets a caller re-run this against a growing array (e.g. one
    more file appended each time) and only look for a run starting within
    the newly-revealed region, rather than re-flagging something already
    checked in an earlier call."""
    rolling = np.asarray(rolling_mean(signal, window_size=window_size))
    above = (rolling > threshold).astype(int)
    if sustained_for <= 1:
        candidates = np.where(above[offset:])[0]
        return int(offset + candidates[0]) if len(candidates) else None
    if len(above) < sustained_for:
        return None
    # window_sums[i] = above[i:i + sustained_for].sum(), via a cumulative-sum
    # trick instead of an explicit sliding window.
    cumsum = np.cumsum(np.insert(above, 0, 0))
    window_sums = cumsum[sustained_for:] - cumsum[:-sustained_for]
    candidates = np.where(window_sums[offset:] == sustained_for)[0]
    if len(candidates) == 0:
        return None
    return int(offset + candidates[0])


class ThresholdDriftDetector(BaseDriftDetector):
    """Original notebook logic: fit() computes `multiplier * max(rolling_mean
    (reference_signal))` as a fixed threshold; run() reports the first index
    where a (possibly growing) signal's rolling mean crosses it and stays
    above it for `sustained_for` consecutive samples (default 1 -- a single
    crossing is enough, the original notebook behavior).

    Low-level primitive: operates on a raw numeric signal array the caller
    already computed, not on a model/inputs/targets. ModelUncertaintyDriftDetector
    and ModelErrorDriftDetector wrap an instance of this to do the signal
    computation for you -- use those (registry ids "model_uncertainty" /
    "model_error") rather than this directly in a CombinedDriftDetector,
    whose kwargs-forwarding convention this class's fit()/run() signatures
    don't match."""

    def __init__(self, rolling_window=50, multiplier=1.25, hard_threshold=None, sustained_for=1, **config):
        super().__init__(rolling_window=rolling_window, multiplier=multiplier, sustained_for=sustained_for, **config)
        self.rolling_window = rolling_window
        self.multiplier = multiplier
        self.threshold = None
        self.hard_threshold = hard_threshold  # optional hard ceiling on threshold
        self.sustained_for = sustained_for

    def fit(self, reference_signal):
        rolling = rolling_mean(reference_signal, window_size=self.rolling_window)
        valid = rolling[~np.isnan(rolling)]
        self.threshold = self.multiplier * float(np.max(valid))
        if self.hard_threshold is not None:
            self.threshold = min(self.threshold, self.hard_threshold)
        return self.threshold

    def run(self, signal, offset=0):
        if self.threshold is None:
            raise RuntimeError("ThresholdDriftDetector.fit(reference_signal) must be called before run().")
        return _first_crossing(signal, self.threshold, self.rolling_window, offset=offset, sustained_for=self.sustained_for)
