import pytest

from l3kit.drift_detectors.base import BaseDriftDetector


def test_base_drift_detector_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseDriftDetector()


def test_subclass_without_run_cannot_be_instantiated():
    class Incomplete(BaseDriftDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()
