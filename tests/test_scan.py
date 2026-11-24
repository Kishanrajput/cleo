import numpy as np
import pytest

from l3kit import DriftNotFoundError, scan_stream
from l3kit.drift_detectors import build_combined_detector


class _SignalDetector:
    """Fires at the first row where inputs[:, 0] > 5 (at/after offset)."""

    def run(self, offset=0, inputs=None, **kwargs):
        hits = np.where(inputs[offset:, 0] > 5)[0]
        return int(offset + hits[0]) if len(hits) else None


def _chunks(values_per_chunk, start_ts=0):
    ts = start_ts
    for i, vals in enumerate(values_per_chunk):
        n = len(vals)
        yield i, np.array(vals, dtype=float).reshape(-1, 1), np.zeros((n, 1)), np.arange(ts, ts + n)
        ts += n


def _score(x, y, ts):
    return np.zeros(len(x)), np.full(len(x), 0.1)


def test_scan_stream_finds_drift_with_global_index_and_chunk_id():
    res = scan_stream(_SignalDetector(), _chunks([[0, 0, 0], [0, 9, 0], [9]]), _score)
    assert res.drift_index == 4 and res.drift_chunk == 1
    assert res.chunks_scanned == 2 and not res.reached_stop
    assert res.inputs.shape == (6, 1) and res.drift_timestamp == 4


def test_scan_stream_raises_after_max_chunks():
    with pytest.raises(DriftNotFoundError):
        scan_stream(_SignalDetector(), _chunks([[0]] * 10), _score, max_chunks=3)


def test_scan_stream_stop_at_timestamp_returns_reached_stop():
    res = scan_stream(_SignalDetector(), _chunks([[0, 0], [0, 0], [9, 9]]), _score, stop_at_timestamp=3)
    assert res.reached_stop and res.drift_index is None and res.chunks_scanned == 2


def test_scan_stream_on_chunk_hook_called_per_chunk():
    seen = []
    scan_stream(_SignalDetector(), _chunks([[0], [9]]), _score, on_chunk=lambda cid, *a: seen.append(cid))
    assert seen == [0, 1]


def test_scan_stream_with_real_combined_detector_reports_thresholds():
    class _Model:
        def run(self, x):
            return np.zeros(len(x)), np.full(len(x), 0.1)

    det = build_combined_detector(["model_uncertainty"], rolling_window=2, sustained_for=1)
    det.fit(model=_Model(), train_inputs=np.zeros((10, 1)), train_targets=np.zeros((10, 1)))

    def score(x, y, ts):  # uncertainty jumps once the input signal does
        return np.zeros(len(x)), np.where(x[:, 0] > 5, 0.19, 0.1)

    res = scan_stream(det, _chunks([[0] * 4, [0, 9, 9, 9]]), score, model=_Model(), max_chunks=5)
    assert res.drift_index is not None and res.drift_index >= 4
    assert res.thresholds["model_uncertainty"] == pytest.approx(0.125)
    assert res.triggered == {"model_uncertainty": True}
