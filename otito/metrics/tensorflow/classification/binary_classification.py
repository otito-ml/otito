import tensorflow as tf

from otito.metrics.tensorflow.base_tensorflow_metric import TensorflowBaseMetric

from otito.metrics.tensorflow.validation.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)
from otito.metrics._base_metric import BaseMetric


class BinaryAccuracy(TensorflowBaseMetric, BaseMetric):
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
    num_correct: tf.Variable
    total: tf.Variable

    def __init__(self, *args, **kwargs):
        TensorflowBaseMetric.__init__(self, *args, name="binary_a", dtype=tf.float32)
        BaseMetric.__init__(
            self, *args, val_config=self.input_validator_config, **kwargs
        )
        self.num_correct = self.add_weight(
            name="num_correct", initializer="zeros", dtype=tf.float32
        )
        self.total = self.add_weight(
            name="total", initializer="zeros", dtype=tf.float32
        )

        if not self.stateful:
            self.__call__ = self.call_metric_function

    def reset(self):
        self.num_correct.assign(0)
        self.total.assign(0)

    def _update_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor
    ) -> float:
        self.num_correct.assign_add(
            tf.reduce_sum(self._tensor_equality(y_observed, y_predicted))
        )
        self.total.assign_add(tf.cast(tf.size(y_observed), dtype=tf.float32))

    def _update_weighted_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor, sample_weights: tf.Tensor
    ) -> float:
        self.num_correct.assign_add(
            tf.tensordot(
                self._tensor_equality(y_observed, y_predicted),
                sample_weights,
                axes=1,
            )
        )
        self.total.assign(1)

    @tf.function
    def tf_print(self, value):
        tf.print(value)

    def update(
        self,
        y_observed: tf.Tensor = None,
        y_predicted: tf.Tensor = None,
        sample_weight: tf.Tensor = None,
    ):
        y_predicted = tf.cast(tf.where(y_predicted >= 0.5, 1.0, 0.0), tf.float32)
        if sample_weight is None:
            self._update_binary_accuracy(y_observed=y_observed, y_predicted=y_predicted)

        else:
            self._update_weighted_binary_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weight,
            )

    def compute(self):
        return tf.math.divide_no_nan(self.num_correct, self.total)
