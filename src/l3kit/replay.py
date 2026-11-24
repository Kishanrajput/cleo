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

    def add_task(
        self, inputs, targets, scores, task_id=None, regime_boundaries=None, context_cols=None, context_bins=4,
    ):
        """Select representative rows from this task and add them to the
        buffer. `scores` (e.g. per-sample absolute error) determines the
        quantile bins; `task_id` defaults to how many tasks have been added
        so far (0-indexed).

        This task's rows can optionally be split into groups before
        stratifying -- EITHER chronologically via `regime_boundaries` OR by
        one or more context variables' own values via `context_cols`/
        `context_bins`, never both (raises ValueError if both are given).
        With neither (the common case), this reduces to plain whole-task
        quantile stratification, same as before either param existed. With
        groups, the total `n_bins * samples_per_bin` budget is split across
        groups proportional to each group's share of this task's rows, and
        each group is independently quantile-stratified over its own
        `n_bins` bins (scaled down to its quota) -- so one task still
        contributes ~n_bins*samples_per_bin rows total, drawn from every
        group it actually contains.

        regime_boundaries: sorted row indices (into `inputs`/`targets`/
        `scores`, 0-based, NOT including 0) where this task's data switches
        to a different internally-detected (chronological) regime (see
        RegimeAwareTrainValSplitPreprocessor's `train_regime_boundaries`
        return value -- pass that here, not the original window-level
        `regime_boundaries`, since this task's rows are the train split
        only). Produces CONTIGUOUS groups.

        context_cols: column indices into `inputs` (0-based) of one or more
        "context" variables (e.g. a beam-configuration PV) whose own values
        -- not `scores` -- should additionally stratify sampling, so every
        quantile-bin of score is represented from every context-value
        regime, not just whichever context value happens to dominate this
        task's rows. Each column is independently quantile-binned into
        `context_bins` bins (same edge convention `_select_indices` uses for
        `scores`); with more than one column, rows are grouped by the JOINT
        combination of each column's own bin (collapses to a single
        column's own bins when only one is given). Produces generally
        NON-contiguous groups. `context_bins` is ignored when `context_cols`
        is None."""
        if task_id is None:
            task_id = len(self._task_inputs)
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        scores = np.asarray(scores)

        if regime_boundaries and context_cols:
            raise ValueError(
                "add_task accepts regime_boundaries or context_cols, not both -- chronological "
                "regime splitting and context-variable grouping are mutually exclusive per call"
            )

        if context_cols:
            groups = self._context_groups(inputs, context_cols, context_bins)
        else:
            boundaries = sorted(regime_boundaries or [])
            edges = [0] + boundaries + [len(scores)]
            groups = [np.arange(start, end) for start, end in zip(edges[:-1], edges[1:]) if start < end]

        indices = self._select_grouped_indices(scores, groups)
        self._task_inputs.append(inputs[indices])
        self._task_targets.append(targets[indices])
        self._task_ids.append(np.full(len(indices), task_id))

    def _select_grouped_indices(self, scores, groups):
        """Shared by every add_task grouping path (none/regime/context):
        split the total n_bins*samples_per_bin quota across `groups`
        (arbitrary, possibly non-contiguous row-index arrays) proportional
        to each group's share of `scores`' rows, independently
        quantile-stratify each group via _select_indices, and remap each
        group's locally selected indices back to row indices into the
        original arrays."""
        total_quota = self.n_bins * self.samples_per_bin
        n = len(scores)
        selected = []
        for group_idx in groups:
            quota = round(total_quota * len(group_idx) / n)
            local_indices = self._select_indices(scores[group_idx], quota)
            selected.extend(group_idx[local_indices].tolist())
        return np.array(selected, dtype=int)

    def _context_groups(self, inputs, context_cols, context_bins):
        """Non-contiguous row groups for context-variable-based replay
        stratification: every one of the given `inputs` columns (0-based
        indices) is independently quantile-binned into `context_bins` bins
        (same edge convention as _select_indices, via _quantile_bin_index);
        rows are then grouped by the JOINT combination of each column's own
        bin id. A single context_cols entry collapses to that column's own
        quantile bins; several combine jointly (via np.unique over the
        stacked per-column bin ids), which only ever produces combinations
        that at least one row actually falls into -- never an empty
        cross-product cell."""
        context_values = np.asarray(inputs)[:, context_cols]
        per_col_bins = np.stack(
            [self._quantile_bin_index(context_values[:, c], context_bins) for c in range(context_values.shape[1])],
            axis=1,
        )
        _, group_ids = np.unique(per_col_bins, axis=0, return_inverse=True)
        group_ids = group_ids.reshape(-1)  # defensive: shape with axis= has varied across NumPy versions
        return [np.where(group_ids == g)[0] for g in range(group_ids.max() + 1)]

    def _quantile_bin_index(self, values, n_bins):
        """Per-row integer bin id (0..n_bins-1) from quantile-binning
        `values` into n_bins bins: first bin is [edges[0], edges[1]]
        inclusive on both ends, every later bin is (edges[j], edges[j+1]]
        -- shared by _select_indices (bins by score) and _context_groups
        (bins by a context variable's own value) so both use the identical
        convention."""
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
        bin_idx = np.full(len(values), -1, dtype=int)
        for j in range(n_bins):
            if j == 0:
                mask = (values >= edges[0]) & (values <= edges[1])
            else:
                mask = (values > edges[j]) & (values <= edges[j + 1])
            bin_idx[mask] = j
        return bin_idx

    def _select_indices(self, scores, quota):
        """Quantile-stratify `scores` into self.n_bins bins and draw up to
        `quota` rows total (samples/bin = round(quota / n_bins), each bin
        still capped at its own size) -- quota is n_bins*samples_per_bin for
        a single-regime task, or a group's proportional share of that when
        called per-group from add_task."""
        if len(scores) == 0 or quota <= 0:
            return np.array([], dtype=int)
        samples_per_bin = max(1, round(quota / self.n_bins))
        bin_idx = self._quantile_bin_index(scores, self.n_bins)
        selected = []
        for j in range(self.n_bins):
            bin_indices = np.where(bin_idx == j)[0]
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
