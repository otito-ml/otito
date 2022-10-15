from math import isclose
import tensorflow as tf

from pydantic import validator


@validator("y_predicted")
def labels_must_be_same_shape(cls, v, values):
    if v.shape != values.get("y_observed").shape:
        raise ValueError(
            f"Shape of inputs mismatched: "
            f"{values.get('y_observed').shape.as_list()[0]} "
            f"(observed) != {v.shape.as_list()[0]} (predicted)"
        )
    return v


@validator("y_predicted")
def labels_must_be_binary(cls, v, values):
    found_classes = tf.unique(tf.concat([v, values.get("y_observed")], 0))[
        0
    ].shape.as_list()[0]
    if found_classes > 2:
        raise ValueError(f"Input is not binary: '{found_classes}' class labels found")
    return v


@validator("sample_weight")
def sample_weight_must_be_same_len(cls, v, values):
    if v is not None:
        if v.shape.as_list()[0] != values.get("y_observed").shape.as_list()[0]:
            raise ValueError(
                "'sample_weight' is not the same length as input. "
                f"Lengths (sample_weight:{v.shape.as_list()[0]}, "
                f"input:{values.get('y_observed').shape.as_list()[0]})"
            )
    return v


@validator("sample_weight")
def sample_weight_must_sum_to_one(cls, v):
    if v is not None:
        if not isclose(tf.math.reduce_sum(v).numpy(), 1.0, abs_tol=1e-7):
            raise ValueError(
                "'sample_weight' do not sum to one. "
                f"Sum of `sample_weight`:{tf.math.reduce_sum(v).numpy()}"
            )
    return v
