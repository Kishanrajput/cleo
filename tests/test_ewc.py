import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from l3kit.ewc import compute_fisher_importance


class _ToyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x), None


def test_compute_fisher_importance_is_nonnegative_and_nonzero_when_gradients_flow():
    torch.manual_seed(0)
    model = _ToyModule()
    x = torch.randn(20, 2)
    y = torch.randn(20, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=5)

    importance = compute_fisher_importance(model, loader)

    assert set(importance.keys()) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert all((v >= 0).all() for v in importance.values())
    assert any(v.sum().item() > 0 for v in importance.values())


def test_compute_fisher_importance_does_not_leave_stale_gradients_on_the_model():
    torch.manual_seed(0)
    model = _ToyModule()
    loader = DataLoader(TensorDataset(torch.randn(10, 2), torch.randn(10, 1)), batch_size=5)

    compute_fisher_importance(model, loader)

    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())
