import pytest
from otito.metrics.numpy import Accuracy


class TestAccuracy:
    """
    Class to test Numpy Accuracy Metric Function
    """

    @pytest.fixture
    def accuracy(self):
        return Accuracy()

    @pytest.mark.usefixtures("accuracy")
    @pytest.mark.parametrize(
        "y_observed,y_predicted,expected",
        [
            ([0, 0, 0, 0], [1, 1, 1, 1], 0),
            ([1, 1, 1, 1], [0, 0, 0, 0], 0),
            ([1, 1, 1, 1], [1, 1, 1, 1], 1),
            ([0, 0, 0, 0], [0, 0, 0, 0], 1),
            ([1, 1, 0, 1], [1, 0, 0, 1], 0.75),
            ([False, False, False, False], [True, True, True, True], 0),
            ([True, True, True, True], [False, False, False, False], 0),
            ([True, True, True, True], [True, True, True, True], 1),
            ([False, False, False, False], [False, False, False, False], 1),
            ([True, True, False, True], [True, False, False, True], 0.75),
        ],
    )
    def test_base_accuracy(self, accuracy, y_observed, y_predicted, expected):
        actual = accuracy(y_observed=y_observed, y_predicted=y_predicted)
        assert expected == pytest.approx(actual)

    @pytest.mark.usefixtures("accuracy")
    @pytest.mark.parametrize(
        "y_observed,y_predicted,sample_weights,expected",
        [
            ([1, 1, 0, 1], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.75),
            ([0, 0, 0, 0], [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], 0.0),
            (
                [True, True, False, True],
                [True, False, False, True],
                [0.25, 0.25, 0.25, 0.25],
                0.75,
            ),
            (
                [False, False, False, False],
                [True, True, True, True],
                [0.25, 0.25, 0.25, 0.25],
                0.0,
            ),
        ],
    )
    def test_weighted_accuracy(
        self, accuracy, y_observed, y_predicted, sample_weights, expected
    ):
        actual = accuracy(
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weights=sample_weights,
        )
        assert expected == pytest.approx(actual)
