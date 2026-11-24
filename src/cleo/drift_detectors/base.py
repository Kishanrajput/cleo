from abc import ABC, abstractmethod


class BaseDriftDetector(ABC):
    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute this DriftDetector's core behavior. Must be overridden."""
        raise NotImplementedError
