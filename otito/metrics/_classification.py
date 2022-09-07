import numpy as np

from otito.metrics.utils import call_metric
from otito.validation.utils import argument_validator
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


def accuracy_score(
    y_observed: np.ndarray,
    y_predicted: np.ndarray,
    sample_weights: np.ndarray = None,
    parse_input: bool = True,
) -> float:
    if sample_weights is not None:
        return call_metric(
            func=_weighted_accuracy,
            validate=parse_input,
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weights=sample_weights,
        )

    else:
        return call_metric(
            func=_base_accuracy,
            validate=parse_input,
            y_observed=y_observed,
            y_predicted=y_predicted,
        )
