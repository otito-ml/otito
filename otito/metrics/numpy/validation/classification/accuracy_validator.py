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
    y_true: Array[float]
    y_pred: Array[float]
    sample_weights: Array[float] = None

    @root_validator
    def input_must_be_same_shape(cls, values):
        if not (len(values.get("y_true")) == len(values.get("y_pred"))):
            raise ValueError(
                f"Shape of inputs mismatched: "
                f"{len(values.get('y_true'))} "
                f"(left) != {len(values.get('y_pred'))} (right)"
            )
        return values
