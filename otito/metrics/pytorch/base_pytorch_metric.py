from abc import ABC

import torch as pt

from otito.metrics._base_metric import StatelessMetricMixin


class PyTorchBaseMetric(StatelessMetricMixin, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    @staticmethod
    def _tensor_equality(left_tensor: pt.Tensor, right_tensor: pt.Tensor) -> pt.Tensor:
        return (left_tensor == right_tensor).float()
