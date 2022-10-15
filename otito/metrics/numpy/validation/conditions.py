from math import isclose
import numpy as np

from pydantic import validator


@validator("y_predicted")
def labels_must_be_same_shape(cls, v, values):
    if v.shape != values.get("y_observed").shape:
        raise ValueError(
            f"Shape of inputs mismatched: "
            f"{values.get('y_observed').shape[0]} "
            f"(observed) != {v.shape[0]} (predicted)"
        )
    return v


@validator("y_predicted")
def labels_must_be_binary(cls, v, values):
    found_classes = np.unique(np.concatenate((v, values.get("y_observed"))))
    if len(found_classes) > 2:
        raise ValueError(
            f"Input is not binary: '{len(found_classes)}' class labels found"
        )
    return v


@validator("sample_weight")
def sample_weight_must_be_same_len(cls, v, values):
    if v is not None:
        if len(v) != len(values.get("y_observed")):
            raise ValueError(
                "'sample_weight' is not the same length as input. "
                f"Lengths (sample_weight:{len(v)}, "
                f"input:{len(values.get('y_observed'))})"
            )
    return v


@validator("sample_weight")
def sample_weight_must_sum_to_one(cls, v):
    if v is not None:
        if not isclose(sum(v), 1.0, abs_tol=1e-7):
            raise ValueError(
                "'sample_weight' do not sum to one. " f"Sum of `sample_weight`:{sum(v)}"
            )
    return v
