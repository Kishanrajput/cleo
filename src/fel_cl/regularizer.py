import torch


class QuadraticPenaltyRegularizer:
    """Shared penalty machinery for EWC and Synaptic Intelligence: both
    reduce to an importance-weighted L2 penalty pulling each parameter back
    toward a reference value, 0.5 * lam * sum(importance * (theta - ref)^2).
    They differ only in how `importance`/`reference` get updated after each
    task -- see ewc.py's compute_fisher_importance and
    synaptic_intelligence.py's SIWeightTracker.

    Online variant: importance accumulates as a single running sum across
    all tasks seen so far (not one stored penalty per task), and reference
    is always the most recently completed task's weights. Constant memory
    regardless of how many tasks have been seen."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.importance = {}
        self.reference = {}

    def penalty(self, model):
        if not self.importance:
            return torch.zeros((), device=next(model.parameters()).device)
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        for name, param in model.named_parameters():
            if name in self.importance:
                total = total + (self.importance[name].to(device) * (param - self.reference[name].to(device)).pow(2)).sum()
        return 0.5 * self.lam * total

    def accumulate(self, task_importance, model):
        """Online update: add this task's importance to the running total,
        move the reference to this task's (just-trained) weights."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            added = task_importance.get(name, torch.zeros_like(param))
            self.importance[name] = self.importance.get(name, torch.zeros_like(param)) + added
        self.reference = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }
