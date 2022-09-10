import numpy as np

from otito.metrics.utils import call_metric
from otito.metrics.numpy.validation.utils import argument_validator
from otito.metrics.numpy.validation.accuracy_validator import (
    BaseAccuracyValidator,
    WeightedAccuracyValidator,
)
from otito.metrics._base_metric import BaseMetric


@argument_validator(BaseAccuracyValidator)
def _base_accuracy(y_observed: np.ndarray, y_predicted: np.ndarray) -> float:
    return np.average(y_observed == y_predicted)


@argument_validator(WeightedAccuracyValidator)
def _weighted_accuracy(
    y_observed: np.ndarray, y_predicted: np.ndarray, sample_weights: np.ndarray
) -> float:
    return np.dot((y_observed == y_predicted), sample_weights)


class Accuracy(BaseMetric):
    def __init__(self, parse_input: bool = True):
        self.parse_input = parse_input

    def calculate(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> float:
        if sample_weights is not None:
            return call_metric(
                func=_weighted_accuracy,
                validate=self.parse_input,
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        else:
            return call_metric(
                func=_base_accuracy,
                validate=self.parse_input,
                y_observed=y_observed,
                y_predicted=y_predicted,
            )

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)
