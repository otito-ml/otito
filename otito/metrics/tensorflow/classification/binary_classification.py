import tensorflow as tf

from otito.base.tensorflow.base_tensorflow_metric import BinaryAccuracyBase
from otito.input_validation.tensorflow.base_validator import BaseTensorflowValidator
from otito.input_validation.tensorflow.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weight_must_be_same_len,
    sample_weight_must_sum_to_one,
)


class BinaryAccuracy(BinaryAccuracyBase):
    """
    The Tensorflow Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

    input_validator_config = {
        "y_observed": (tf.Tensor, None),
        "y_predicted": (tf.Tensor, None),
        "sample_weight": (tf.Tensor, None),
        "__validators__": {
            "labels_must_be_same_shape": labels_must_be_same_shape,
            "labels_must_be_binary": labels_must_be_binary,
            "sample_weight_must_be_same_len": sample_weight_must_be_same_len,
            "sample_weight_must_sum_to_one": sample_weight_must_sum_to_one,
        },
        "__base__": BaseTensorflowValidator,
    }

    num_correct = None
    total = None
    threshold = None

    def initialise_states(self, threshold=0.5):
        self.num_correct = self.add_weight(
            name="num_correct", initializer="zeros", dtype=tf.float32
        )
        self.total = self.add_weight(
            name="total", initializer="zeros", dtype=tf.float32
        )
        self.threshold = threshold

    def _update_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor
    ) -> float:
        self.num_correct.assign_add(
            tf.reduce_sum(self._tensor_equality(y_observed, y_predicted))
        )
        self.total.assign_add(tf.cast(tf.size(y_observed), dtype=tf.float32))

    def _update_weighted_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor, sample_weight: tf.Tensor
    ) -> float:
        self.num_correct.assign_add(
            tf.tensordot(
                self._tensor_equality(y_observed, y_predicted),
                sample_weight,
                axes=1,
            )
        )
        self.total.assign(1)

    def reset(self):
        self.num_correct.assign(0)
        self.total.assign(0)

    def update(
        self,
        y_observed: tf.Tensor = None,
        y_predicted: tf.Tensor = None,
        sample_weight: tf.Tensor = None,
    ):
        y_predicted = self.apply_threshold(y_predicted, threshold=self.threshold)

        if sample_weight is None:
            self._update_binary_accuracy(y_observed=y_observed, y_predicted=y_predicted)

        else:
            self._update_weighted_binary_accuracy(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weight=sample_weight,
            )

    def compute(self):
        return tf.math.divide_no_nan(self.num_correct, self.total)
