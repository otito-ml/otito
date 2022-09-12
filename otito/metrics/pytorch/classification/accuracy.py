import torch as pt


from otito.metrics.pytorch.validation.classification.accuracy_validator import (
    BaseAccuracyValidator,
)
from otito.metrics._base_metric import BaseMetric, PyTorchBaseMetric
from otito.metrics.utils import argument_validator


class Accuracy(PyTorchBaseMetric, BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _base_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor
    ) -> pt.Tensor:
        return (y_observed == y_predicted).float().mean(dim=0)

    def _weighted_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weights: pt.Tensor
    ) -> pt.Tensor:
        return pt.dot((y_observed == y_predicted).float(), sample_weights)

    @argument_validator(BaseAccuracyValidator)
    def compute(
        self,
        y_observed: pt.Tensor = None,
        y_predicted: pt.Tensor = None,
        sample_weights: pt.Tensor = None,
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

    @argument_validator(BaseAccuracyValidator)
    def update(
        self,
        y_observed: pt.Tensor = None,
        y_predicted: pt.Tensor = None,
        sample_weights: pt.Tensor = None,
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
