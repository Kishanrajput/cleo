from l3kit.drift_detectors.combined_detector import CombinedDriftDetector


class _FixedIndexDetector:
    def __init__(self, index):
        self.index = index
        self.fit_kwargs = None
        self.run_kwargs = None

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs

    def run(self, offset=0, **kwargs):
        self.run_kwargs = kwargs
        return self.index


class _NeverFiresDetector:
    def run(self, offset=0, **kwargs):
        return None


def test_run_returns_the_earliest_triggered_index():
    detector = CombinedDriftDetector({"a": _FixedIndexDetector(10), "b": _FixedIndexDetector(5)})

    idx = detector.run(inputs=[0.0] * 20)

    assert idx == 5
    assert detector.triggered == {"a": True, "b": True}


def test_run_returns_none_when_nothing_triggers():
    detector = CombinedDriftDetector({"a": _NeverFiresDetector(), "b": _NeverFiresDetector()})

    idx = detector.run(inputs=[0.0] * 20)

    assert idx is None
    assert detector.triggered == {"a": False, "b": False}


def test_triggered_records_which_sub_detector_actually_fired():
    detector = CombinedDriftDetector({"threshold": _NeverFiresDetector(), "adwin": _FixedIndexDetector(7)})

    idx = detector.run(inputs=[0.0] * 20)

    assert idx == 7
    assert detector.triggered == {"threshold": False, "adwin": True}


def test_run_forwards_the_same_kwargs_to_every_sub_detector():
    a = _FixedIndexDetector(None)
    b = _FixedIndexDetector(None)
    detector = CombinedDriftDetector({"a": a, "b": b})

    detector.run(offset=5, model="the-model", inputs="the-inputs", targets="the-targets")

    assert a.run_kwargs == {"model": "the-model", "inputs": "the-inputs", "targets": "the-targets"}
    assert b.run_kwargs == a.run_kwargs


def test_fit_forwards_kwargs_to_every_sub_detector_that_has_a_fit_method():
    class _NoFitDetector:
        def run(self, offset=0, **kwargs):
            return None

    a = _FixedIndexDetector(None)
    no_fit = _NoFitDetector()
    detector = CombinedDriftDetector({"a": a, "no_fit": no_fit})

    detector.fit(model="the-model", train_inputs="the-train-inputs")  # must not raise on the no-fit detector

    assert a.fit_kwargs == {"model": "the-model", "train_inputs": "the-train-inputs"}
