from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

    def call_metric(self, func, validate=True, *args, **kwargs):
        if validate:
            return func(*args, **kwargs)
        else:
            return func.__wrapped__(self, *args, **kwargs)
