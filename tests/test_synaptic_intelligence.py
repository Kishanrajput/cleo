import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fel_cl.regularizer import QuadraticPenaltyRegularizer
from fel_cl.synaptic_intelligence import SIWeightTracker


class _ToyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x), None

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred, _ = self(x)
        return F.mse_loss(pred, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_si_weight_tracker_finalize_populates_regularizer_after_training(tmp_path):
    torch.manual_seed(0)
    model = _ToyModule()
    loader = DataLoader(TensorDataset(torch.randn(20, 2), torch.randn(20, 1)), batch_size=5)

    tracker = SIWeightTracker(xi=1e-3)
    trainer = L.Trainer(
        max_epochs=1, callbacks=[tracker], enable_progress_bar=False, enable_checkpointing=False,
        logger=False, accelerator="cpu", default_root_dir=str(tmp_path),
    )
    trainer.fit(model, train_dataloaders=loader)

    regularizer = QuadraticPenaltyRegularizer(lam=1.0)
    tracker.finalize(model, regularizer)

    param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(regularizer.importance.keys()) == param_names
    assert set(regularizer.reference.keys()) == param_names
    for name, param in model.named_parameters():
        assert torch.equal(regularizer.reference[name], param.detach())
