import inspect
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, parse_input=True, *args, **kwargs):
        self.parse_input = parse_input
        self.callable = self.compute

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def merge_args_kwargs(self, *args, **kwargs):
        arg_names = inspect.getfullargspec(self.callable.__wrapped__).args
        arg_names.remove("self")
        kwargs.update(dict(zip(arg_names, args)))
        return kwargs

    def call_metric(self, *args, **kwargs):
        if self.parse_input:
            return self.callable(*args, **kwargs)
        else:
            return self.callable.__wrapped__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        metric_arguments = self.merge_args_kwargs(*args, **kwargs)
        return self.call_metric(**metric_arguments)


class PyTorchBaseMetric(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callable = self.update

    @abstractmethod
    def update(self, *arg, **kwargs):
        pass
