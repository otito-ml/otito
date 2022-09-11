import importlib
import functools


def load_metric(
    metric: str = "Accuracy", package: str = "numpy", parse_input: bool = True
):
    metric_module = importlib.import_module(name=f"otito.metrics.{package}")
    callable_metric = getattr(metric_module, metric)
    return callable_metric(parse_input=parse_input)


def argument_validator(validator_model):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            validated_kwargs = validator_model(**kwargs)
            return func(*args, **validated_kwargs.dict())

        return inner_wrapper

    return outer_wrapper
