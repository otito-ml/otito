from pydantic import BaseModel as BasePyVal
import torch as pt


class BaseAccuracyValidator(BasePyVal):
    y_true: pt.Tensor = None
    y_pred: pt.Tensor = None
    sample_weights: pt.Tensor = None

    class Config:
        arbitrary_types_allowed = True
