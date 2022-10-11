from abc import ABC

import numpy as np

from otito.metrics._base_metric import BaseMetric


class NumpyBaseMetric(BaseMetric, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    @staticmethod
    def _array_equality(
        left_tensor: np.ndarray, right_tensor: np.ndarray
    ) -> np.ndarray:
        return (left_tensor == right_tensor).astype(float)

    @BaseMetric.validation_handler
    def call_metric_function(
        self,
        y_observed: np.ndarray,
        y_predicted: np.ndarray,
        sample_weights: np.ndarray,
    ):
        self.reset()
        self.update(
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weights=sample_weights,
        )
        return self.compute()
