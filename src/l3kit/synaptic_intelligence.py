import lightning as L
import torch


class SIWeightTracker(L.Callback):
    """Synaptic Intelligence (Zenke et al. 2017) online importance
    accumulation: at every optimizer step, tracks
    w[p] += -grad[p] * delta[p] (the per-parameter contribution to the drop
    in loss along the optimization path). At finalize(), normalizes by each
    parameter's total movement over the task
    (importance = w / (total_delta^2 + xi)) and folds it into a shared
    regularizer.QuadraticPenaltyRegularizer via its online accumulate().

    Call finalize() (or compute_importance(), see below) right after
    trainer.fit() returns, BEFORE restoring best-checkpoint weights -- the
    importance is a property of the full optimization trajectory, not any
    single epoch's weights, so it should be computed from the trajectory's
    actual endpoint."""

    def __init__(self, xi=1e-3):
        self.xi = xi
        self._w = {}
        self._prev_params = {}
        self._theta_start = {}
        self._grads = {}

    def on_train_start(self, trainer, pl_module):
        params = {name: param for name, param in pl_module.named_parameters() if param.requires_grad}
        self._w = {name: torch.zeros_like(param) for name, param in params.items()}
        self._theta_start = {name: param.detach().clone() for name, param in params.items()}
        self._prev_params = {name: param.detach().clone() for name, param in params.items()}

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._grads = {
            name: (param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param))
            for name, param in pl_module.named_parameters() if param.requires_grad
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            delta = param.detach() - self._prev_params[name]
            self._w[name] += -self._grads.get(name, torch.zeros_like(param)) * delta
            self._prev_params[name] = param.detach().clone()

    def compute_importance(self, model):
        """Per-parameter SI importance from this task's full optimization
        trajectory (see class docstring), independent of any regularizer --
        e.g. so a training run with no CL regularizer active yet (base-model
        training) can still compute and save this trajectory's importance to
        disk, for a LATER continual-learning run to accumulate into its own
        regularizer as task 0's real SI seed, instead of falling back to an
        EWC-style post-hoc approximation (see
        ContinualLearningDriver.run())."""
        task_importance = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # `model` may have been moved back to a different device (e.g.
            # Lightning returns the module to CPU after trainer.fit()
            # completes) than the one it was on when self._theta_start /
            # self._w were captured -- realign before combining them.
            theta_start = self._theta_start[name].to(param.device)
            w = self._w[name].to(param.device)
            total_delta = param.detach() - theta_start
            task_importance[name] = w / (total_delta.pow(2) + self.xi)
        return task_importance

    def finalize(self, model, regularizer):
        regularizer.accumulate(self.compute_importance(model), model)
