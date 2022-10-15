import inspect
import importlib
import functools


def load_metric(
    metric: str = "BinaryAccuracy", package: str = "numpy", *args, **kwargs
):
    metric_module = importlib.import_module(name=f"otito.metrics.{package}")
    callable_metric = getattr(metric_module, metric)
    kwargs.update({"package": package})
    return callable_metric(*args, **kwargs)


def argument_validator(validator_model):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            validated_kwargs = validator_model(**kwargs)
            return func(*args, **validated_kwargs.dict())

        return inner_wrapper

    return outer_wrapper


def validation_handler(func):
    @functools.wraps(func)
    def inner(self, *args, **kwargs):
        return func(self, **self._parse_input(*args, **kwargs))

    return inner


def get_function_arg_names(function):
    arg_names = inspect.getfullargspec(function).args
    if "self" in arg_names:
        arg_names.remove("self")
    return arg_names
