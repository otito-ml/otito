from abc import ABC

import tensorflow as tf

from otito.metrics._base_metric import BaseMetric


class TensorflowBaseMetric(BaseMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    @staticmethod
    def _tensor_equality(left_tensor: tf.Tensor, right_tensor: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.math.equal(left_tensor, right_tensor), tf.float32)
