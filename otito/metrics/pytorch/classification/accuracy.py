import torch as pt

from otito.metrics._base_metric import BaseMetric


def _base_accuracy(y_observed: pt.Tensor, y_predicted: pt.Tensor) -> pt.Tensor:
    return (y_observed == y_predicted).float().mean(dim=0)


def _weighted_accuracy(
    y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weights: pt.Tensor
) -> pt.Tensor:
    return pt.dot((y_observed == y_predicted).float(), sample_weights)


class Accuracy(BaseMetric):
    def __init__(self):
        pass

    def calculate(
        self,
        y_observed: pt.Tensor,
        y_predicted: pt.Tensor,
        sample_weights: pt.Tensor = None,
    ) -> float:
        if sample_weights is not None:
            return _weighted_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        else:
            return _base_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
            )

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)
