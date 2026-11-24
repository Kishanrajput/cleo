import numpy as np


class ReplayBuffer:
    """Growing, quantile-stratified replay buffer.

    Implements Algorithm 1 ("Replay Sample Selection with Quantile-Based
    Sampling") from Rajput et al., "Continual Learning for Particle
    Accelerators" (NeurIPS ML4PS 2025), generalized from 1 sample per
    quantile bin to `samples_per_bin` samples per bin, and specifically the
    paper's "growing replay" variant: buffer size is unbounded (no analog of
    the paper's `if |B| > M: resample` pruning step -- M is effectively
    infinite here, exactly as in the paper's own growing-replay experiment).

    Each call to add_task() selects up to `n_bins * samples_per_bin`
    representative rows from that task's own data, stratified by whatever
    per-sample score (e.g. absolute error) is passed in, and appends them to
    the buffer tagged with that task's id -- the buffer grows by one task's
    worth of samples every time.
    """

    def __init__(self, n_bins=20, samples_per_bin=100, rng=None):
        self.n_bins = n_bins
        self.samples_per_bin = samples_per_bin
        self.rng = rng if rng is not None else np.random.default_rng()
        self._task_inputs = []
        self._task_targets = []
        self._task_ids = []

    def add_task(self, inputs, targets, scores, task_id=None, regime_boundaries=None):
        """Select representative rows from this task and add them to the
        buffer. `scores` (e.g. per-sample absolute error) determines the
        quantile bins; `task_id` defaults to how many tasks have been added
        so far (0-indexed).

        regime_boundaries: sorted row indices (into `inputs`/`targets`/
        `scores`, 0-based, NOT including 0) where this task's data switches
        to a different internally-detected regime (see
        RegimeAwareTrainValSplitPreprocessor's `train_regime_boundaries`
        return value -- pass that here, not the original window-level
        `regime_boundaries`, since this task's rows are the train split
        only). None or [] (the common case, one regime) reduces to plain
        whole-task quantile stratification, same as before this param
        existed. With multiple regimes, the total `n_bins * samples_per_bin`
        budget is split across regimes proportional to each regime's share
        of this task's rows, and each regime is independently
        quantile-stratified over its own `n_bins` bins (scaled down to its
        quota) -- so one task still contributes ~n_bins*samples_per_bin rows
        total, drawn from every regime it actually contains rather than only
        the chronologically-last one."""
        if task_id is None:
            task_id = len(self._task_inputs)
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        scores = np.asarray(scores)
        total_quota = self.n_bins * self.samples_per_bin

        boundaries = sorted(regime_boundaries or [])
        edges = [0] + boundaries + [len(scores)]
        selected = []
        for start, end in zip(edges[:-1], edges[1:]):
            if start >= end:
                continue
            quota = round(total_quota * (end - start) / len(scores))
            regime_indices = self._select_indices(scores[start:end], quota)
            selected.extend((regime_indices + start).tolist())

        indices = np.array(selected, dtype=int)
        self._task_inputs.append(inputs[indices])
        self._task_targets.append(targets[indices])
        self._task_ids.append(np.full(len(indices), task_id))

    def _select_indices(self, scores, quota):
        """Quantile-stratify `scores` into self.n_bins bins and draw up to
        `quota` rows total (samples/bin = round(quota / n_bins), each bin
        still capped at its own size) -- quota is n_bins*samples_per_bin for
        a single-regime task, or a regime's proportional share of that when
        called per-regime from add_task."""
        if len(scores) == 0 or quota <= 0:
            return np.array([], dtype=int)
        samples_per_bin = max(1, round(quota / self.n_bins))
        edges = np.quantile(scores, np.linspace(0.0, 1.0, self.n_bins + 1))
        selected = []
        for j in range(self.n_bins):
            if j == 0:
                bin_mask = (scores >= edges[0]) & (scores <= edges[1])
            else:
                bin_mask = (scores > edges[j]) & (scores <= edges[j + 1])
            bin_indices = np.where(bin_mask)[0]
            if len(bin_indices) == 0:
                continue
            k = min(samples_per_bin, len(bin_indices))
            selected.extend(self.rng.choice(bin_indices, size=k, replace=False).tolist())
        return np.array(selected, dtype=int)

    def get_replay_data(self):
        """(inputs, targets, task_ids) concatenated across every task added
        so far, or (None, None, None) if nothing has been added yet."""
        if not self._task_inputs:
            return None, None, None
        inputs = np.concatenate(self._task_inputs, axis=0)
        targets = np.concatenate(self._task_targets, axis=0)
        task_ids = np.concatenate(self._task_ids, axis=0)
        return inputs, targets, task_ids

    def __len__(self):
        return sum(len(arr) for arr in self._task_inputs)


def task_inverse_weights(task_ids):
    """Per-row sampling weight = 1 / (number of rows sharing that task id),
    so a WeightedRandomSampler represents every task roughly equally per
    epoch regardless of how many rows it actually contributes."""
    task_ids = np.asarray(task_ids)
    _, inverse, counts = np.unique(task_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def make_replay_loader(inputs, targets, batch_size, task_ids=None):
    """Training DataLoader over (inputs, targets). With `task_ids` (one per
    row, e.g. new-task rows concatenated with a ReplayBuffer's rows), rows are
    drawn by a WeightedRandomSampler with task_inverse_weights so each task is
    represented roughly equally per epoch; without, plain shuffling."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    if task_ids is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    weights = task_inverse_weights(task_ids)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
