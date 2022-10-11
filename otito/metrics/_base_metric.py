import inspect
import functools
from abc import ABC, abstractmethod

from pydantic import create_model


class BaseMetric(ABC):
    def __init__(self, *args, **kwargs):
        self.callable = self.compute
        self.validate_input = kwargs["validate_input"]
        self.validator = self._build_validator(kwargs["package"], kwargs["val_config"])

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def _build_validator(self, package, config):
        return create_model(f"{package}:{self.__class__.__name__}Model", **config)

    def _merge_args_kwargs(self, *args, **kwargs):
        arg_names = inspect.getfullargspec(self.callable).args
        arg_names.remove("self")
        kwargs.update(dict(zip(arg_names, args)))
        return kwargs

    def _parse_input(self, *args, **kwargs):
        metric_arguments = self._merge_args_kwargs(*args, **kwargs)
        if self.validate_input:
            metric_arguments = self.validator(**metric_arguments).dict()
        else:
            metric_arguments = self.validator.construct(**metric_arguments).dict()
        return metric_arguments

    def __call__(self, *args, **kwargs):
        return self.callable(**self._parse_input(*args, **kwargs))


class PyTorchBaseMetric(BaseMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callable = self.call_metric_function
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    def validation_handler(func):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return func(self, **self._parse_input(*args, **kwargs))

        return inner
