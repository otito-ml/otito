import torch as pt

from otito.metrics.pytorch.base_pytorch_metric import PyTorchBaseMetric
from otito.metrics.pytorch.validation.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class BinaryAccuracy(PyTorchBaseMetric):
    """
    The Pytorch Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

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

    correct: pt.Tensor
    total: pt.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    def reset(self):
        self.correct: pt.Tensor = pt.tensor(0.0, dtype=pt.float32)
        self.total: float = 0.0

    def _update_binary_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor
    ) -> pt.Tensor:
        self.correct += pt.sum(self._tensor_equality(y_observed, y_predicted))
        self.total += y_observed.numel()

    def _update_weighted_binary_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weights: pt.Tensor
    ) -> pt.Tensor:
        self.correct += pt.sum(
            pt.dot(self._tensor_equality(y_observed, y_predicted), sample_weights)
        )
        self.total = 1.0

    def update(
        self,
        y_observed: pt.Tensor = None,
        y_predicted: pt.Tensor = None,
        sample_weights: pt.Tensor = None,
    ):
        if sample_weights is None:
            self._update_binary_accuracy(y_observed=y_observed, y_predicted=y_predicted)

        else:
            self._update_weighted_binary_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

    def compute(self) -> float:
        return self.correct.float() / self.total
