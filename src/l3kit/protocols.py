from typing import Protocol, Tuple

import numpy as np


class Predictor(Protocol):
    """What the model-based drift detectors (model_error, model_uncertainty)
    need from a model: `run(inputs) -> (predictions, std)`."""

    def run(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
