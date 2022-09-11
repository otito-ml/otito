from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, parse_input=True):
        self.parse_input = parse_input

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

    def call_metric(self, function, validate=True, *args, **kwargs):
        if validate:
            return function(*args, **kwargs)
        else:
            return function.__wrapped__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_metric(
            function=self.calculate, validate=self.parse_input, *args, **kwargs
        )
