from math import isclose
import torch as pt

from pydantic import validator


@validator("y_predicted")
def labels_must_be_same_shape(cls, v, values):
    if v.shape != values.get("y_observed").shape:
        raise ValueError(
            f"Shape of inputs mismatched: "
            f"{values.get('y_observed').size()[0]} "
            f"(observed) != {v.size()[0]} (predicted)"
        )
    return v


@validator("y_predicted")
def labels_must_be_binary(cls, v, values):
    found_classes = pt.unique(pt.cat((v, values.get("y_observed")), 0))
    if list(found_classes.size())[0] > 2:
        raise ValueError(
            f"Input is not binary: '{found_classes.size()[0]}' class labels found"
        )
    return v


@validator("sample_weights")
def sample_weights_must_be_same_len(cls, v, values):
    if v is not None:
        if v.shape[0] != values.get("y_observed").shape[0]:
            raise ValueError(
                "'sample_weights' is not the same length as input. "
                f"Lengths (sample_weights:{len(v)}, "
                f"input:{len(values.get('y_observed'))})"
            )
    return v


@validator("sample_weights")
def sample_weights_must_sum_to_one(cls, v):
    if v is not None:
        if not isclose(v.sum().item(), 1.0, abs_tol=1e-7):
            raise ValueError(
                "'sample_weights' do not sum to one. "
                f"Sum of `sample_weights`:{v.sum().item()}"
            )
    return v
