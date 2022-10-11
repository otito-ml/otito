import tensorflow as tf
from tensorflow.keras.metrics import Metric as KerasMetric

from otito.metrics._base_metric import BaseMetric

from otito.metrics.tensorflow.validation.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class BinaryAccuracy(BaseMetric, KerasMetric):
    """
    The Tensorflow Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

    class Config:
        arbitrary_types_allowed = True

    input_validator_config = {
        "y_observed": (tf.Tensor, None),
        "y_predicted": (tf.Tensor, None),
        "sample_weights": (tf.Tensor, None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weights_must_be_same_len": sample_weights_must_be_same_len,
            "sample_weights_must_sum_to_one": sample_weights_must_sum_to_one,
        },
        "__config__": Config,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    @staticmethod
    def _base_accuracy(y_observed: tf.Tensor, y_predicted: tf.Tensor) -> float:
        return tf.math.reduce_mean(
            tf.cast(tf.math.equal(y_observed, y_predicted), tf.float32)
        )

    @staticmethod
    def _weighted_accuracy(
        y_observed: tf.Tensor, y_predicted: tf.Tensor, sample_weights: tf.Tensor
    ) -> float:
        return tf.tensordot(
            tf.cast(tf.math.equal(y_observed, y_predicted), tf.float32),
            sample_weights,
            axes=1,
        )

    def compute(
        self,
        y_observed: tf.Tensor = None,
        y_predicted: tf.Tensor = None,
        sample_weights: tf.Tensor = None,
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

    def update_state(self):
        pass

    def result(self):
        pass

    def reset_state(self):
        pass
