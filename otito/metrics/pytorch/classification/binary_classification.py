import torch as pt

from otito.base.pytorch.base_pytorch_metric import BinaryAccuracyBase
from otito.input_validation.pytorch.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weight_must_be_same_len,
    sample_weight_must_sum_to_one,
)


class BinaryAccuracy(BinaryAccuracyBase):
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
        "sample_weight": (pt.Tensor, None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weight_must_be_same_len": sample_weight_must_be_same_len,
            "sample_weight_must_sum_to_one": sample_weight_must_sum_to_one,
        },
        "__config__": Config,
    }

    num_correct = None
    total = None
    threshold = None

    full_state_update = False

    def initialise_states(self, threshold=0.5):
        self.add_state("correct", default=pt.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=pt.tensor(0.0), dist_reduce_fx="sum")
        self.threshold = threshold

    def _update_binary_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor
    ) -> pt.Tensor:
        self.correct += pt.sum(self._tensor_equality(y_observed, y_predicted)).type(
            pt.int32
        )
        self.total += y_observed.numel()

    def _update_weighted_binary_accuracy(
        self, y_observed: pt.Tensor, y_predicted: pt.Tensor, sample_weight: pt.Tensor
    ) -> pt.Tensor:
        self.correct += pt.dot(
            self._tensor_equality(y_observed, y_predicted), sample_weight
        )
        self.total = pt.tensor(1.0)

    def update(
        self,
        y_observed: pt.Tensor = None,
        y_predicted: pt.Tensor = None,
        sample_weight: pt.Tensor = None,
    ):
        y_predicted = self.apply_threshold(y_predicted, self.threshold)

        if sample_weight is None:
            self._update_binary_accuracy(y_observed=y_observed, y_predicted=y_predicted)

        else:
            self._update_weighted_binary_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weight=sample_weight,
            )

    def compute(self) -> float:
        return self.correct.float() / self.total
