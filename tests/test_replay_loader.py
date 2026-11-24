import numpy as np

from l3kit import ReplayBuffer, make_replay_loader, task_inverse_weights


def test_task_inverse_weights_balance_tasks():
    w = task_inverse_weights([0, 0, 0, 0, 1])
    assert np.allclose(w, [0.25, 0.25, 0.25, 0.25, 1.0])


def test_make_replay_loader_plain_and_weighted():
    x, y = np.random.rand(10, 3), np.random.rand(10, 1)
    plain = make_replay_loader(x, y, batch_size=4)
    assert sum(len(b[0]) for b in plain) == 10
    weighted = make_replay_loader(x, y, batch_size=4, task_ids=[0] * 9 + [1])
    assert sum(len(b[0]) for b in weighted) == 10


def test_replay_buffer_importable_from_package_root():
    assert len(ReplayBuffer()) == 0


def test_task_inverse_weights_balances_unequal_task_sizes():
    w = task_inverse_weights(np.array([0] * 900 + [1] * 100))
    assert np.allclose(w[:900], 1.0 / 900) and np.allclose(w[900:], 1.0 / 100)
    assert np.isclose(w[:900].sum(), w[900:].sum())
