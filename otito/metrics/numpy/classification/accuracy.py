import numpy as np

from otito.metrics import BaseMetric
from otito.metrics.numpy.validation.accuracy_validator import (
    BaseAccuracyValidator,
    WeightedAccuracyValidator,
)
from otito.metrics.utils import argument_validator


class Accuracy(BaseMetric):
    def __init__(self, parse_input: bool = True):
        self.parse_input = parse_input

    @argument_validator(BaseAccuracyValidator)
    def _base_accuracy(self, y_observed: np.ndarray, y_predicted: np.ndarray) -> float:
        return np.average(y_observed == y_predicted)

    @argument_validator(WeightedAccuracyValidator)
    def _weighted_accuracy(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        return np.dot((y_observed == y_predicted), sample_weights)

    def calculate(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> float:
        if sample_weights is not None:
            return self.call_metric(
                func=self._weighted_accuracy,
                validate=self.parse_input,
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        else:
            return self.call_metric(
                func=self._base_accuracy,
                validate=self.parse_input,
                y_observed=y_observed,
                y_predicted=y_predicted,
            )

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)
