import torch as pt

from otito.metrics._base_metric import BaseMetric
from otito.metrics.pytorch.validation.classification.accuracy_validator import (
    BaseAccuracyValidator,
)
from otito.metrics.utils import argument_validator


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _base_accuracy(self, y_true: pt.Tensor, y_pred: pt.Tensor) -> pt.Tensor:
        return (y_true == y_pred).float().mean(dim=0)

    def _weighted_accuracy(
        self, y_true: pt.Tensor, y_pred: pt.Tensor, sample_weights: pt.Tensor
    ) -> pt.Tensor:
        return pt.dot((y_true == y_pred).float(), sample_weights)

    @argument_validator(BaseAccuracyValidator)
    def calculate(
        self,
        y_true: pt.Tensor,
        y_pred: pt.Tensor,
        sample_weights: pt.Tensor = None,
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
