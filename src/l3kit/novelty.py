import numpy as np


def fit_novelty_reference(values):
    """Per-column mean/std of a reference (training) set, used by
    novelty_score. Computed once on the base model's training inputs (for
    input-space OOD) or training targets (for output-space OOD)."""
    values = np.asarray(values)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1e-8, std)
    return mean, std


def novelty_score(values, ref_mean, ref_std):
    """Per-row scalar novelty score: root-mean-square standardized distance
    of each row from a reference distribution's mean/std (a diagonal-
    covariance Mahalanobis-style distance -- full covariance isn't used
    since inverting a high-dimensional covariance matrix is both expensive
    and numerically fragile at typical sample sizes here). Works on any
    per-row numeric feature vector -- input features (many columns) or a
    single output/target column -- one scalar per row either way, suitable
    for feeding a single ADWIN instance instead of monitoring every
    variable individually."""
    values = np.asarray(values)
    standardized = (values - ref_mean) / ref_std
    return np.sqrt(np.mean(standardized ** 2, axis=1))
