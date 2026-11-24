"""l3kit: generic, model-agnostic continual-learning building blocks.

Replay buffers, quadratic-penalty regularizers (EWC, Synaptic Intelligence),
drift detectors, forgetting metrics and a data-agnostic stream scanner. No
dependency on any particular dataset, model class or experiment framework.
"""
from .replay import ReplayBuffer, task_inverse_weights, make_replay_loader
from .regularizer import QuadraticPenaltyRegularizer
from .ewc import compute_fisher_importance
from .synaptic_intelligence import SIWeightTracker
from .forgetting import evaluate_on_past_tasks, forgetting_metrics
from .scan import scan_stream, StreamScanResult, DriftNotFoundError
from .protocols import Predictor

__all__ = [
    "ReplayBuffer", "task_inverse_weights", "make_replay_loader", "QuadraticPenaltyRegularizer",
    "compute_fisher_importance", "SIWeightTracker", "evaluate_on_past_tasks", "forgetting_metrics",
    "scan_stream", "StreamScanResult", "DriftNotFoundError", "Predictor",
]
