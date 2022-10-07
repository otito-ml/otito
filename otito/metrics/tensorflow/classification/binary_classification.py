import tensorflow as tf

from otito.metrics._base_metric import BaseMetric

# from otito.metrics.tensorflow.validation.conditions import


class BinaryAccuracy(BaseMetric):
    """
    The Pytorch Binary Classification Accuracy Metric provides a score that
    represents the proportion of a dataset that was correctly labeled by a
    binary classifier
    """

    class Config:
        arbitrary_types_allowed = True

    input_validator_config = {
        "y_observed": (tf.Tensor, None),
        "y_predicted": (tf.Tensor, None),
        "sample_weights": (tf.Tensor, None),
        "__config__": Config,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, val_config=self.input_validator_config, **kwargs)

    def _base_accuracy(self, y_observed: tf.Tensor, y_predicted: tf.Tensor) -> float:
        return tf.math.reduce_mean(tf.cast((y_observed == y_predicted), tf.float32))

    def _weighted_accuracy(
        self, y_observed: tf.Tensor, y_predicted: tf.Tensor, sample_weights: tf.Tensor
    ) -> float:
        return tf.tensordot(
            tf.cast((y_observed == y_predicted), tf.float32), sample_weights, axes=1
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
