import tensorflow as tf

from otito.metrics.tensorflow.base_tensorflow_metric import TensorflowBaseMetric

from otito.metrics.tensorflow.validation.conditions import (
    labels_must_be_same_shape,
    labels_must_be_binary,
    sample_weights_must_be_same_len,
    sample_weights_must_sum_to_one,
)


class BinaryAccuracy(TensorflowBaseMetric):
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

    correct: tf.Tensor
    total: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    def reset(self):
        self.correct: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
        self.total: float = 0.0

    def _update_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor
    ) -> float:
        self.correct += tf.reduce_sum(self._tensor_equality(y_observed, y_predicted))
        self.total += tf.size(y_observed).numpy()

    def _update_weighted_binary_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor, sample_weights: tf.Tensor
    ) -> float:
        self.correct += tf.tensordot(
            self._tensor_equality(y_observed, y_predicted),
            sample_weights,
            axes=1,
        )
        self.total = 1.0

    def update(
        self,
        y_observed: tf.Tensor = None,
        y_predicted: tf.Tensor = None,
        sample_weights: tf.Tensor = None,
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
        return self.correct.numpy() / self.total
