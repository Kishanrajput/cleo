from cleo.drift_detectors.debounced_detector import DebouncedDriftDetector


class _ScriptedDetector:
    """Returns a pre-scripted sequence of run() results, one per call
    (index or None), so a test can drive exactly the fire/quiet pattern it
    wants to check the debounce logic against."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def run(self, offset=0, **kwargs):
        result = self.results[self.calls]
        self.calls += 1
        return result


def test_does_not_confirm_on_a_single_isolated_hit():
    # Fires once (a one-off spike) then goes quiet for good -- exactly the
    # false-positive pattern debounce exists to filter.
    inner = _ScriptedDetector([None, 10, None, None, None, None])
    detector = DebouncedDriftDetector(inner, confirm_for=2, lookback=6)

    results = [detector.run() for _ in range(6)]

    assert all(r is None for r in results)


def test_confirms_once_it_fires_a_second_time_within_the_lookback_window():
    # Fires, goes quiet (ADWIN settling into the new regime), fires again --
    # matches how a real, persistent shift actually behaves with ADWIN,
    # unlike requiring back-to-back consecutive fires.
    inner = _ScriptedDetector([None, 5, None, None, 12, None])
    detector = DebouncedDriftDetector(inner, confirm_for=2, lookback=6)

    results = [detector.run() for _ in range(6)]

    # Confirmed on the call where the 2nd hit lands (index 4), reporting
    # the FIRST index the inner detector ever reported (5), not the 2nd (12).
    assert results[4] == 5
    assert results[:4] == [None, None, None, None]


def test_confirms_immediately_when_confirm_for_is_one():
    inner = _ScriptedDetector([None, 7])
    detector = DebouncedDriftDetector(inner, confirm_for=1, lookback=6)

    assert detector.run() is None
    assert detector.run() == 7


def test_resets_after_the_whole_lookback_window_goes_quiet():
    # One hit, then enough quiet calls to fully flush the lookback window
    # (size 3: needs 3 quiet calls to evict the stale hit entirely), then a
    # later unrelated cluster of 2 hits -- must confirm on the LATER
    # cluster's first index, not resurrect the long-stale isolated one.
    inner = _ScriptedDetector([20, None, None, None, 30, None, 31])
    detector = DebouncedDriftDetector(inner, confirm_for=2, lookback=3)

    results = [detector.run() for _ in range(7)]

    assert results == [None, None, None, None, None, None, 30]


def test_fit_forwards_to_the_wrapped_detector_when_it_has_one():
    class _WithFit:
        def __init__(self):
            self.fit_kwargs = None

        def fit(self, **kwargs):
            self.fit_kwargs = kwargs

        def run(self, offset=0, **kwargs):
            return None

    inner = _WithFit()
    detector = DebouncedDriftDetector(inner)

    detector.fit(train_inputs="x")

    assert inner.fit_kwargs == {"train_inputs": "x"}


def test_fit_is_a_noop_when_wrapped_detector_has_no_fit():
    class _NoFit:
        def run(self, offset=0, **kwargs):
            return None

    detector = DebouncedDriftDetector(_NoFit())
    detector.fit(train_inputs="x")  # must not raise
