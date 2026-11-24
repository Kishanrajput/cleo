import lightning as L
import torch
import torch.nn as nn

from fel_cl.regularizer import QuadraticPenaltyRegularizer


class _ToyModule(L.LightningModule):
    """Minimal LightningModule -- keeps these tests fast and dependency-free
    (no real GPA layer needed)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x), None


def test_quadratic_penalty_regularizer_is_zero_with_no_importance():
    model = _ToyModule()
    reg = QuadraticPenaltyRegularizer(lam=1.0)

    assert torch.isclose(reg.penalty(model), torch.zeros(()))


def test_quadratic_penalty_regularizer_is_zero_at_the_reference_point():
    model = _ToyModule()
    reg = QuadraticPenaltyRegularizer(lam=2.0)
    reg.reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    reg.importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    assert torch.isclose(reg.penalty(model), torch.zeros(()))


def test_quadratic_penalty_regularizer_grows_with_deviation_from_reference():
    model = _ToyModule()
    reg = QuadraticPenaltyRegularizer(lam=2.0)
    reg.reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    reg.importance = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    with torch.no_grad():
        model.linear.weight += 1.0
    penalty = reg.penalty(model)

    assert penalty.item() > 0


def test_quadratic_penalty_regularizer_accumulate_sums_importance_online():
    model = _ToyModule()
    reg = QuadraticPenaltyRegularizer()

    task1_importance = {n: torch.full_like(p, 2.0) for n, p in model.named_parameters()}
    reg.accumulate(task1_importance, model)
    first_reference = {n: p.clone() for n, p in reg.reference.items()}

    with torch.no_grad():
        model.linear.weight += 1.0
        model.linear.bias += 1.0
    task2_importance = {n: torch.full_like(p, 3.0) for n, p in model.named_parameters()}
    reg.accumulate(task2_importance, model)

    for n, p in model.named_parameters():
        assert torch.allclose(reg.importance[n], torch.full_like(p, 5.0))  # 2.0 + 3.0, running sum
        assert torch.equal(reg.reference[n], p.detach())  # reference moved to latest task's weights
        assert not torch.equal(reg.reference[n], first_reference[n])
