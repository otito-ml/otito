from math import isclose
from pydantic import BaseModel as BasePyVal, root_validator

from otito.metrics.numpy.validation.custom_types import Array


class BaseAccuracyValidator(BasePyVal):
    y_observed: Array[float]
    y_predicted: Array[float]
    sample_weights: Array[float] = None

    @root_validator
    def input_must_be_same_shape(cls, values):
        if not (len(values.get("y_observed")) == len(values.get("y_predicted"))):
            raise ValueError(
                f"Shape of inputs mismatched: "
                f"{len(values.get('y_observed'))} "
                f"(left) != {len(values.get('y_predicted'))} (right)"
            )
        return values

    @root_validator
    def input_must_be_binary(cls, values):
        max_size = max(
            (len(set(values.get("y_observed"))), "y_observed"),
            (len(set(values.get("y_predicted"))), "y_predicted"),
        )
        if max_size[0] > 2:
            raise ValueError(
                f"Input is not binary: '{max_size[1]}' "
                f"contains {max_size[0]} classes. "
                f" Classes found in '{max_size[1]}': {set(values.get(max_size[1]))}"
            )
        return values

    @root_validator
    def weights_must_be_same_length_as_samples(cls, values):
        if values.get("sample_weights") is not None:
            if len(values.get("y_observed")) != len(values.get("sample_weights")):
                raise ValueError(
                    "'sample_weights' is not the same length as input. "
                    f"Lengths (sample_weights:{len(values.get('sample_weights'))}, "
                    f"input:{len(values.get('y_observed'))})"
                )
        return values

    @root_validator
    def weights_must_sum_to_one(sel, values):
        if values.get("sample_weights") is not None:
            if not isclose(sum(values.get("sample_weights")), 1.0, abs_tol=1e-7):
                raise ValueError(
                    "'sample_weights' do not sum to one. "
                    f"Sum of `sample_weights`:{sum(values.get('sample_weights'))}"
                )
        return values
