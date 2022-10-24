import numpy as np

from otito.base.numpy.base_numpy_metric import NumpyBaseMetric
from otito.input_validation.numpy.custom_types import Array
from otito.input_validation.numpy.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weight_must_be_same_len,
    sample_weight_must_sum_to_one,
)


class BinaryAccuracy(NumpyBaseMetric):
    """
    The Numpy Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

    input_validator_config = {
        "y_observed": (Array[float], None),
        "y_predicted": (Array[float], None),
        "sample_weight": (Array[float], None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weight_must_be_same_len": sample_weight_must_be_same_len,
            "sample_weight_must_sum_to_one": sample_weight_must_sum_to_one,
        },
    }

    correct: int
    total: int

    def __init__(self, *args, **kwargs):
        """
        This is a test docstring
        :param args:
        :param kwargs:

        """
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    def reset(self):
        self.correct = 0
        self.total = 0

    def initialise_states(self):
        self.correct = 0
        self.total = 0

    def _update_binary_accuracy(
        self, y_observed: np.ndarray, y_predicted: np.ndarray
    ) -> float:
        self.correct += np.sum(self._array_equality(y_observed, y_predicted))
        self.total += y_observed.size

    def _update_weighted_binary_accuracy(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        self.correct += np.dot(
            self._array_equality(y_observed, y_predicted), sample_weight
        )
        self.total = 1.0

    def update(
        self,
        y_observed: np.ndarray = None,
        y_predicted: np.ndarray = None,
        sample_weight: np.ndarray = None,
    ):
        if sample_weight is None:
            self._update_binary_accuracy(y_observed=y_observed, y_predicted=y_predicted)

        else:
            self._update_weighted_binary_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weight=sample_weight,
            )

    def compute(self) -> float:
        return self.correct / self.total
