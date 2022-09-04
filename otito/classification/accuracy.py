import numpy as np

from otito.validation.utils import argument_validator, get_metric
from otito.validation.base_validators import (
    BaseAccuracyValidator,
    WeightedAccuracyValidator,
)


@argument_validator(BaseAccuracyValidator)
def _base_accuracy(y_observed: np.ndarray, y_predicted: np.ndarray) -> float:
    return np.average(y_observed == y_predicted)


@argument_validator(WeightedAccuracyValidator)
def _weighted_accuracy(
    y_observed: np.ndarray, y_predicted: np.ndarray, sample_weights: np.ndarray
) -> float:
    return np.dot((y_observed == y_predicted), sample_weights)


def accuracy(
    y_observed: np.ndarray,
    y_predicted: np.ndarray,
    sample_weights: np.ndarray = None,
    validate_input: bool = True,
) -> float:
    if sample_weights is not None:
        weighted_accuracy = get_metric(_weighted_accuracy, validate_input)
        return weighted_accuracy(
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weights=sample_weights,
        )

    else:
        base_accuracy = get_metric(_base_accuracy, validate_input)
        return base_accuracy(y_observed=y_observed, y_predicted=y_predicted)
