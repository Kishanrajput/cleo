from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class DriftNotFoundError(RuntimeError):
    """max_chunks were scanned without the detector firing."""


@dataclass
class StreamScanResult:
    drift_index: Optional[int]        # row index into the concatenated arrays below; None if no drift
    drift_chunk: Optional[object]     # chunk id (as yielded by `chunks`) the drift fired in
    chunks_scanned: int
    reached_stop: bool                # True if the scan ended at stop_at_timestamp without a hit
    thresholds: dict = field(default_factory=dict)
    triggered: dict = field(default_factory=dict)
    inputs: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    @property
    def drift_timestamp(self):
        if self.drift_index is None or self.timestamps is None:
            return None
        return self.timestamps[self.drift_index]


def detector_thresholds(detector):
    """{name: threshold} for every sub-detector of a CombinedDriftDetector
    that exposes one (threshold-based ones do; ADWIN-based don't)."""
    return {name: sub.threshold for name, sub in getattr(detector, "detectors", {}).items() if hasattr(sub, "threshold")}


def scan_stream(
    detector, chunks, score_fn, model=None, max_chunks=None, stop_at_timestamp=None, on_chunk=None,
):
    """Feed a stream to a fitted drift `detector` chunk by chunk until it fires.

    chunks: iterable of (chunk_id, inputs, targets, timestamps); may be an
        infinite generator (e.g. "the next data file").
    score_fn(inputs, targets, timestamps) -> (predictions, std): scores one
        chunk with whatever model the caller is monitoring.
    model: forwarded to detector.run(model=...) for detectors that need it.
    max_chunks: raise DriftNotFoundError after this many chunks with no hit.
    stop_at_timestamp: rows at/after it are dropped; reaching it without a hit
        returns reached_stop=True instead of raising.
    on_chunk(chunk_id, inputs, targets, predictions, std): optional hook,
        e.g. for live logging.

    The detector sees the growing concatenation of every kept chunk and is
    called with offset=<rows already checked>, so it only looks for drift
    starting inside the newly revealed rows."""
    all_inputs, all_targets, all_preds, all_std, all_ts = [], [], [], [], []
    global_offset = 0
    chunks_scanned = 0

    def finish(drift_index, drift_chunk, reached_stop):
        cat = (lambda xs: np.concatenate(xs) if xs else None)
        return StreamScanResult(
            drift_index=drift_index, drift_chunk=drift_chunk, chunks_scanned=chunks_scanned,
            reached_stop=reached_stop, thresholds=detector_thresholds(detector),
            triggered=dict(getattr(detector, "triggered", {})),
            inputs=cat(all_inputs), targets=cat(all_targets), timestamps=cat(all_ts),
            predictions=cat(all_preds), std=cat(all_std),
        )

    for chunk_id, inputs, targets, timestamps in chunks:
        if max_chunks is not None and chunks_scanned >= max_chunks:
            break

        reached_stop = False
        if stop_at_timestamp is not None:
            keep = timestamps <= stop_at_timestamp
            reached_stop = not bool(keep.all())
            if not keep.any():
                return finish(None, None, True)
            inputs, targets, timestamps = inputs[keep], targets[keep], timestamps[keep]

        preds, std = score_fn(inputs, targets, timestamps)
        if on_chunk is not None:
            on_chunk(chunk_id, inputs, targets, preds, std)
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_preds.append(preds)
        all_std.append(std)
        all_ts.append(timestamps)
        chunks_scanned += 1

        drift_idx = detector.run(
            offset=global_offset, model=model, inputs=np.concatenate(all_inputs),
            targets=np.concatenate(all_targets), preds=np.concatenate(all_preds), std=np.concatenate(all_std),
        )
        if drift_idx is not None:
            return finish(drift_idx, chunk_id, False)
        if reached_stop:
            return finish(None, None, True)
        global_offset += len(inputs)

    raise DriftNotFoundError(f"No drift detected after scanning {chunks_scanned} chunks.")
