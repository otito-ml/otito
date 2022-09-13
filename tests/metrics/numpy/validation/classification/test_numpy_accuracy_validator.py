import pytest
from otito.metrics.numpy.validation.classification import BaseAccuracyValidator


class TestAccuracyValidator:
    """
    Class to test Numpy Accuracy Metric Function
    """

    @pytest.mark.parametrize(
        "y_observed,y_predicted",
        [([1, 1, 1, 1], [1, 1, 1])],
    )
    def test_input_must_be_same_shape(self, y_observed, y_predicted):
        with pytest.raises(ValueError) as _:
            BaseAccuracyValidator(y_observed=y_observed, y_predicted=y_predicted)
