import torch as pt

from otito.metrics._base_metric import BaseMetric
from otito.metrics.pytorch.validation.accuracy_validator import (
    BaseAccuracyValidator,
    WeightedAccuracyValidator,
)
from otito.metrics.utils import argument_validator


class Accuracy(BaseMetric):
    def __init__(self, parse_input):
        self.parse_input = parse_input

    @argument_validator(BaseAccuracyValidator)
    def _base_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor
    ) -> pt.Tensor:
        return (y_observed == y_predicted).float().mean(dim=0)

    @argument_validator(WeightedAccuracyValidator)
    def _weighted_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weights: pt.Tensor
    ) -> pt.Tensor:
        return pt.dot((y_observed == y_predicted).float(), sample_weights)

    def calculate(
        self,
        y_observed: pt.Tensor,
        y_predicted: pt.Tensor,
        sample_weights: pt.Tensor = None,
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
