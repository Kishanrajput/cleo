import numpy as np

from cleo.signal import rolling_mean


def test_rolling_mean_leading_nan_for_incomplete_windows():
    assert np.isnan(rolling_mean([1, 2, 3], window_size=5)).all()
