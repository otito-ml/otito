import inspect
from abc import ABC, abstractmethod

from pydantic import create_model

from otito.metrics.utils import validation_handler


class BaseMetric(ABC):
    def __init__(self, *args, **kwargs):
        self.callable = self.call_metric_function
        self.validate_input = kwargs["validate_input"]
        self.validator = self._build_validator(kwargs["package"], kwargs["val_config"])
        self.metric_args = self.get_metric_args()
        self.stateful = kwargs["stateful"]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def get_metric_args(self):
        arg_names = inspect.getfullargspec(self.update).args
        arg_names.remove("self")
        return arg_names

    def _build_validator(self, package, config):
        return create_model(f"{package}:{self.__class__.__name__}Model", **config)

    def _merge_args_kwargs(self, *args, **kwargs):
        kwargs.update(dict(zip(self.metric_args, args)))
        return kwargs

    def _parse_input(self, *args, **kwargs):
        metric_arguments = self._merge_args_kwargs(*args, **kwargs)
        if self.validate_input:
            metric_arguments = self.validator(**metric_arguments).dict()
        else:
            metric_arguments = self.validator.construct(**metric_arguments).dict()
        return metric_arguments

    @validation_handler
    def call_metric_function(self, **kwargs):
        self.update(**kwargs)
        result = self.compute()

        if not self.stateful:
            self.reset()
        return result

    def __call__(self, *args, **kwargs):
        return self.callable(**self._parse_input(*args, **kwargs))
