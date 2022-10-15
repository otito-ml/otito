from abc import ABC, abstractmethod

from pydantic import create_model

from otito.metrics.utils import get_function_arg_names


class StatelessMetricMixin(ABC):
    def __init__(
        self,
        *args,
        validate_input=None,
        package=None,
        val_config=None,
        stateful=None,
        **kwargs,
    ):
        self.validate_input = validate_input
        self.validator = self._build_validator(package, val_config)
        self.metric_args = get_function_arg_names(self.update)
        self.stateful = stateful

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def _build_validator(self, package, config):
        return create_model(f"{package}:{self.__class__.__name__}Model", **config)

    def _merge_args_kwargs(self, *args, **kwargs):
        kwargs.update(dict(zip(self.metric_args, args)))
        return kwargs

    def _parse_input(self, *args, **kwargs):
        metric_arguments = self._merge_args_kwargs(*args, **kwargs)
        if self.validate_input:
            metric_arguments = self.validator(**metric_arguments).dict()
        return metric_arguments

    def stateless_metric_call(self, **kwargs):
        self.update(**kwargs)
        result = self.compute()
        if not self.stateful:
            self.reset()
        return result

    def __call__(self, *args, **kwargs):
        return self.stateless_metric_call(**self._parse_input(*args, **kwargs))
