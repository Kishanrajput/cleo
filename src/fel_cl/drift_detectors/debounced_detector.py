from collections import deque

from .base import BaseDriftDetector


class DebouncedDriftDetector(BaseDriftDetector):
    """Wraps another detector and only confirms drift once it has fired on
    at least `confirm_for` scan calls within the most recent `lookback`
    calls -- filters an isolated one-file spike while tolerating how a
    stateful detector like ADWIN actually behaves on a real, persistent
    shift: it typically fires once, at the moment of change, then goes
    quiet as its internal window re-settles around the new regime, rather
    than re-firing on every subsequent call. Requiring N *consecutive*
    fires would suppress exactly the real detections this exists to keep;
    requiring N fires within a bounded recent window doesn't.

    Intended for wrapping the ADWIN-based OOD detectors (input/output),
    which are more prone to firing on a single noisy file than the
    threshold-based uncertainty/error detectors."""

    def __init__(self, detector, confirm_for=2, lookback=6, **config):
        super().__init__(detector=detector, confirm_for=confirm_for, lookback=lookback, **config)
        self.detector = detector
        self.confirm_for = confirm_for
        self.lookback = max(lookback, confirm_for)
        self._hits = deque(maxlen=self.lookback)
        self._first_index = None

    def fit(self, **kwargs):
        fit_fn = getattr(self.detector, "fit", None)
        if fit_fn is not None:
            fit_fn(**kwargs)

    def run(self, offset=0, **kwargs):
        idx = self.detector.run(offset=offset, **kwargs)
        self._hits.append(idx)

        if all(hit is None for hit in self._hits):
            self._first_index = None
        elif self._first_index is None:
            self._first_index = next(hit for hit in self._hits if hit is not None)

        n_hits = sum(1 for hit in self._hits if hit is not None)
        if n_hits >= self.confirm_for:
            return self._first_index
        return None
