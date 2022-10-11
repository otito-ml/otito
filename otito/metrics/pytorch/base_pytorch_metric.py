from abc import ABC, abstractmethod


import torch as pt

from otito.metrics._base_metric import BaseMetric


class PyTorchBaseMetric(BaseMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callable = self.call_metric_function
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    def _tensor_equality(left_tensor: pt.Tensor, right_tensor: pt.Tensor) -> pt.Tensor:
        return (left_tensor == right_tensor).float()

    @BaseMetric.validation_handler
    def call_metric_function(self, **kwargs):
        self.reset()
        self.update(**kwargs)
        return self.compute()
