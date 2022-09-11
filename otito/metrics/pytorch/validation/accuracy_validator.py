from pydantic import BaseModel as BasePyVal
import torch as pt


class BaseAccuracyValidator(BasePyVal):
    y_observed: pt.Tensor = None
    y_predicted: pt.Tensor = None

    class Config:
        arbitrary_types_allowed = True


class WeightedAccuracyValidator(BaseAccuracyValidator):
    sample_weights: pt.Tensor = None
