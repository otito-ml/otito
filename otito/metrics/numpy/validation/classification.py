from math import isclose
import numpy as np
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
                f"(observed) != {len(values.get('y_predicted'))} (predicted)"
            )
        return values

    @root_validator
    def input_must_be_binary(cls, values):
        found_classes = set(
            np.concatenate((values.get("y_observed"), values.get("y_predicted")))
        )
        if len(found_classes) > 2:
            raise ValueError(
                f"Input is not binary: '{len(found_classes)}' "
                f"classes found. Found classes: `{list(found_classes)}`"
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
