import numpy as np

from otito.metrics._base_metric import BaseMetric
from otito.metrics.numpy.validation.custom_types import Array
from otito.metrics.numpy.validation.classification.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class Accuracy(BaseMetric):

    input_validator_config = {
        "y_observed": (Array[float], None),
        "y_predicted": (Array[float], None),
        "sample_weights": (Array[float], None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weights_must_be_same_len": sample_weights_must_be_same_len,
            "sample_weights_must_sum_to_one": sample_weights_must_sum_to_one,
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs["val_config"] = self.input_validator_config
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
