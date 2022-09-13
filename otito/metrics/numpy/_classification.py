import numpy as np

from otito.metrics import BaseMetric
from otito.metrics.numpy.validation.classification import (
    BaseAccuracyValidator,
)
from otito.metrics.utils import argument_validator


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _base_accuracy(self, y_observed: np.ndarray, y_predicted: np.ndarray) -> float:
        return np.average(y_observed == y_predicted)

    def _weighted_accuracy(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        return np.dot((y_observed == y_predicted), sample_weights)

    @argument_validator(BaseAccuracyValidator)
    def compute(
        self,
        y_observed: np.ndarray = None,
        y_predicted: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ) -> float:
        if sample_weights is None:
            return self._base_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
            )

        else:
            return self._weighted_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )
