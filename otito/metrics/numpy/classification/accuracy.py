import numpy as np

from otito.metrics import BaseMetric
from otito.metrics.numpy.validation.classification.accuracy_validator import (
    BaseAccuracyValidator,
)
from otito.metrics.utils import argument_validator


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _base_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.average(y_true == y_pred)

    def _weighted_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        return np.dot((y_true == y_pred), sample_weights)

    @argument_validator(BaseAccuracyValidator)
    def calculate(
        self,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ) -> float:
        if sample_weights is None:
            return self._base_accuracy(
                y_true=y_true,
                y_pred=y_pred,
            )

        else:
            return self._weighted_accuracy(
                y_true=y_true,
                y_pred=y_pred,
                sample_weights=sample_weights,
            )
