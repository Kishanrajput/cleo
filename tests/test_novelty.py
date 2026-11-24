import numpy as np

from cleo.novelty import fit_novelty_reference, novelty_score


def test_fit_novelty_reference_recovers_mean_and_std():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(5000, 4))

    mean, std = fit_novelty_reference(data)

    assert np.allclose(mean, 3.0, atol=0.1)
    assert np.allclose(std, 2.0, atol=0.1)


def test_novelty_score_is_zero_at_the_reference_mean():
    ref_mean = np.array([1.0, 2.0, 3.0])
    ref_std = np.array([1.0, 1.0, 1.0])
    rows = np.tile(ref_mean, (5, 1))

    scores = novelty_score(rows, ref_mean, ref_std)

    assert np.allclose(scores, 0.0)


def test_novelty_score_grows_with_distance_from_reference():
    ref_mean = np.zeros(3)
    ref_std = np.ones(3)
    near = np.array([[0.1, 0.1, 0.1]])
    far = np.array([[5.0, 5.0, 5.0]])

    near_score = novelty_score(near, ref_mean, ref_std)[0]
    far_score = novelty_score(far, ref_mean, ref_std)[0]

    assert far_score > near_score


def test_novelty_score_works_on_a_single_column():
    # Output/target OOD detection passes a single-column array -- the RMS
    # over one column should just reduce to the absolute z-score.
    ref_mean = np.array([2.0])
    ref_std = np.array([1.0])
    rows = np.array([[2.0], [4.0], [0.0]])

    scores = novelty_score(rows, ref_mean, ref_std)

    assert np.allclose(scores, [0.0, 2.0, 2.0])


def test_fit_novelty_reference_floors_near_zero_std_to_avoid_division_blowup():
    data = np.column_stack([np.ones(10), np.arange(10)])  # first column has zero variance

    _, std = fit_novelty_reference(data)

    assert std[0] > 0
    scores = novelty_score(data, *fit_novelty_reference(data))
    assert np.all(np.isfinite(scores))
