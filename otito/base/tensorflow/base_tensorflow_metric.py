from abc import ABC

import types
import tensorflow as tf
from keras import backend
from keras.engine import base_layer
from keras.utils import metrics_utils
from keras.dtensor import dtensor_api as dtensor
from keras.utils import tf_utils

from otito.base._base_metric import StatelessMetricMixin


class TensorflowBaseMetric(base_layer.Layer, ABC):
    """
    Wrapper Class for Tensorflow Metrics
    """

    def __init__(self, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.stateful = True
        self.built = True
        self._dtype = tf.as_dtype(dtype).name

    def __new__(cls, *args, **kwargs):
        # TODO investigate if the update function need ever not be tf.function
        new_object = super(TensorflowBaseMetric, cls).__new__(cls)
        obj_update_state = new_object.update_state
        obj_result = new_object.result

        def update_state_fn(*args, **kwargs):
            control_status = tf.__internal__.autograph.control_status_ctx()
            ag_update_state = tf.__internal__.autograph.tf_convert(
                obj_update_state, control_status
            )
            return ag_update_state(*args, **kwargs)

        def result_fn(*args, **kwargs):
            control_status = tf.__internal__.autograph.control_status_ctx()
            ag_result = tf.__internal__.autograph.tf_convert(obj_result, control_status)
            return ag_result(*args, **kwargs)

        new_object.update_state = types.MethodType(
            metrics_utils.update_state_wrapper(update_state_fn), new_object
        )
        new_object.result = types.MethodType(
            metrics_utils.result_wrapper(result_fn), new_object
        )

        return new_object

    def __str__(self):
        args = ",".join(f"{k}={v}" for k, v in self.get_config().items())
        return f"{self.__class__.__name__}({args})"

    @property
    def dtype(self):
        return self._dtype

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype}

    def __deepcopy__(self, memo=None):
        try:
            new_self = self.from_config(self.get_config())
        except NotImplementedError as e:
            raise NotImplementedError(
                "Calling `__deepcopy__()` on a Keras metric "
                "requires the metric to be serializable,  "
                "i.e. it should implement `get_config()`.\n\n"
                f"Error encountered during serialization: [{e}]"
            )
        # Note that metrics don't implement `build()` so their variables
        # are readily available after instantiation.
        if self.weights:
            new_self.set_weights(self.get_weights())
        memo[self] = new_self
        return new_self

    def reset_state(self):
        backend.batch_set_value([(v, 0) for v in self.variables])

    def update_state(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def result(self):
        return self.compute()

    def add_weight(
        self,
        name,
        shape=(),
        aggregation=tf.VariableAggregation.SUM,
        synchronization=tf.VariableSynchronization.ON_READ,
        initializer=None,
        dtype=None,
    ):
        """Adds state variable. Only for use by subclasses."""
        if tf.distribute.has_strategy():
            strategy = tf.distribute.get_strategy()
        else:
            strategy = None

        # TODO(b/120571621): Make `ON_READ` work with Keras metrics on TPU.
        if backend.is_tpu_strategy(strategy):
            synchronization = tf.VariableSynchronization.ON_WRITE
        if getattr(self, "_mesh", None) is not None:
            # When self._mesh is set, it means this metric is used for DTensor.
            additional_kwargs = {
                "layout": dtensor.Layout.replicated(
                    self._mesh, tf.TensorShape(shape).rank
                )
            }
        else:
            additional_kwargs = {}

        with tf_utils.maybe_init_scope(layer=self):
            return super().add_weight(
                name=name,
                shape=shape,
                dtype=self._dtype if dtype is None else dtype,
                trainable=False,
                initializer=initializer,
                collections=[],
                synchronization=synchronization,
                aggregation=aggregation,
                **additional_kwargs,
            )

    @staticmethod
    def _tensor_equality(left_tensor: tf.Tensor, right_tensor: tf.Tensor) -> tf.Tensor:
        return tf.cast(tf.math.equal(left_tensor, right_tensor), tf.float32)

    @staticmethod
    def apply_threshold(predicted_tensor, threshold=0.5):
        return tf.cast(tf.where(predicted_tensor >= threshold, 1.0, 0.0), tf.float32)


class BinaryAccuracyBase(StatelessMetricMixin, TensorflowBaseMetric):
    def __init__(self, *args, name=None, dtype=tf.float32, threshold=0.5, **kwargs):
        TensorflowBaseMetric.__init__(self, *args, name=name, dtype=dtype)
        StatelessMetricMixin.__init__(
            self, *args, val_config=self.input_validator_config, **kwargs
        )
        self.initialise_states(threshold=threshold)

    def reset(self):
        pass

    def compute(self):
        pass

    def update(self):
        pass

    def initialise_states(self):
        pass
