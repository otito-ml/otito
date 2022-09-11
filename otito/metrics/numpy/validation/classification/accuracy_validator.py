from pydantic import BaseModel as BasePyVal, root_validator
import numpy as np


class TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)


class ArrayMeta(type):
    def __getitem__(self, t):
        return type("Array", (TypedArray,), {"inner_type": t})


class Array(np.ndarray, metaclass=ArrayMeta):
    pass


class BaseAccuracyValidator(BasePyVal):
    y_observed: Array[float] = None
    y_predicted: Array[float] = None
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
