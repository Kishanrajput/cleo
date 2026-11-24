import torch
import torch.nn as nn


def compute_fisher_importance(model, dataloader, n_batches=None, loss_fn=None):
    """EWC importance: empirical diagonal Fisher, approximated (standard
    practice for regression) as the average squared gradient of the
    training loss w.r.t. each parameter, evaluated at the model's current
    (post-training) weights over its own task data.

    The model's forward may return a tensor or a tuple whose first element is
    the prediction (extra outputs, e.g. an uncertainty, are ignored).
    `loss_fn(preds, targets)` defaults to MSE."""
    was_training = model.training
    model.eval()  # deterministic forward (no dropout) while still tracking grads
    importance = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
    n_seen = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        if n_batches is not None and batch_idx >= n_batches:
            break
        model.zero_grad(set_to_none=True)
        out = model(x)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        loss = loss_fn(preds, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().pow(2)
        n_seen += 1
    for name in importance:
        importance[name] /= max(n_seen, 1)
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return importance
