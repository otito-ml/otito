__all__ = ["BinaryAccuracy"]

import numpy as np

from otito.metrics._base_metric import BaseMetric
from otito.metrics.numpy.validation.custom_types import Array
from otito.metrics.numpy.validation.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class BinaryAccuracy(BaseMetric):
    """
    The Numpy Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

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
        """
        This is a test docstring
        :param args:
        :param kwargs:

        """
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    def compute(
        self,
        y_observed: np.ndarray = None,
        y_predicted: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ) -> float:
        """
        Return an accuracy Score

        :param y_observed: The observed sample labels.
        :param y_predicted: The predicted sample labels.
        :param sample_weights: Optional The weight given to each sample.

        :type y_observed: numpy.Array[int]
        :type y_predicted: numpy.Array[int]
        :type sample_weights: numpy.Array[int] or None

        :return: The Accuracy Score
        :rtype: float
        """
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

    @staticmethod
    def _base_accuracy(y_observed: np.ndarray, y_predicted: np.ndarray) -> float:
        return np.average(y_observed == y_predicted)

    @staticmethod
    def _weighted_accuracy(
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        return np.dot((y_observed == y_predicted), sample_weights)
