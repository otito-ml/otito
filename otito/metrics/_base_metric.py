from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass
