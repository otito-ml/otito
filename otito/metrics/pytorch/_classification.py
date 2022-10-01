import torch as pt

from otito.metrics._base_metric import BaseMetric, PyTorchBaseMetric
from otito.metrics.pytorch.validation.classification.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class Accuracy(PyTorchBaseMetric, BaseMetric):
    class Config:
        arbitrary_types_allowed = True

    input_validator_config = {
        "y_observed": (pt.Tensor, None),
        "y_predicted": (pt.Tensor, None),
        "sample_weights": (pt.Tensor, None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weights_must_be_same_len": sample_weights_must_be_same_len,
            "sample_weights_must_sum_to_one": sample_weights_must_sum_to_one,
        },
        "__config__": Config,
    }

    def __init__(self, *args, **kwargs):
        kwargs["val_config"] = self.input_validator_config
        super().__init__(*args, **kwargs)

    def _base_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor
    ) -> pt.Tensor:
        return (y_observed == y_predicted).float().mean(dim=0)

    def _weighted_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weights: pt.Tensor
    ) -> pt.Tensor:
        return pt.dot((y_observed == y_predicted).float(), sample_weights)

    @PyTorchBaseMetric.validation_handler
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
