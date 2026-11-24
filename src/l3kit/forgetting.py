import numpy as np


def evaluate_on_past_tasks(model, task_val_data, current_cycle, metric_fn):
    """Metric of `model` on every task's val data from BEFORE `current_cycle`
    (task `current_cycle` itself is skipped -- its entry is the diagonal the
    adaptation step already computed). Must not mutate the model.

    task_val_data: {task_index: (val_inputs, val_targets)}.
    metric_fn(model, inputs, targets) -> float (e.g. MAE); injected so this
    module stays independent of any particular evaluator or model class.
    Returns {task_index: metric}."""
    results = {}
    for j in range(current_cycle):
        past_inputs, past_targets = task_val_data[j]
        results[j] = metric_fn(model, past_inputs, past_targets)
    return results


def forgetting_metrics(matrix, lower_is_better=True):
    """Standard continual-learning summary of a forgetting matrix, where
    matrix[i, j] = metric of the model right after training on task i,
    evaluated on task j (NaN where j > i / not yet created).

    Returns a dict:
      final_average:      mean over tasks of the last row's metric.
      average_forgetting: mean over tasks j < T-1 of
                          (final metric on j) - (best earlier metric on j)
                          for lower-is-better metrics (sign flipped otherwise);
                          >= 0 means the task got worse after being learned.
      backward_transfer:  mean over j < T-1 of (diagonal j) - (final metric j),
                          sign-adjusted so positive = the task improved.
    Needs at least 2 tasks; with one task, forgetting/BWT are None."""
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    sign = 1.0 if lower_is_better else -1.0
    out = {"final_average": float(np.nanmean(m[-1, :n]))}
    if n < 2:
        out.update(average_forgetting=None, backward_transfer=None)
        return out
    final = m[-1, : n - 1]
    best_earlier = np.array([np.nanmin(sign * m[j : n - 1, j]) * sign for j in range(n - 1)])
    out["average_forgetting"] = float(np.mean(sign * (final - best_earlier)))
    diagonal = np.diag(m)[: n - 1]
    out["backward_transfer"] = float(np.mean(-sign * (final - diagonal)))
    return out


def plot_forgetting_heatmap(forgetting_matrix, output_dir, metric_name="MAE"):
    """matrix[i, j] = metric of the model as it stood right after adaptation i,
    evaluated on task j's validation data (NaN where j hadn't been created yet).
    Row 0 is the base model. Writes forgetting_heatmap.png. Requires
    matplotlib (install `l3kit[plots]`)."""
    import os

    import matplotlib.pyplot as plt

    n = forgetting_matrix.shape[0]
    masked = np.ma.masked_invalid(forgetting_matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 1.2 * n + 2))
    im = ax.imshow(masked, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"task {j}" for j in range(n)])
    ax.set_yticks(range(n))
    ax.set_yticklabels(["base model"] + [f"after adapt {i}" for i in range(1, n)])
    ax.set_xlabel("Evaluated on task's validation data")
    ax.set_ylabel("Model state")
    ax.set_title(f"Catastrophic forgetting: {metric_name} per task across adaptation cycles")
    for i in range(n):
        for j in range(n):
            if not np.isnan(forgetting_matrix[i, j]):
                ax.text(j, i, f"{forgetting_matrix[i, j]:.4f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label=metric_name)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "forgetting_heatmap.png"))
    plt.close(fig)
