import pytest


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y_observed,y_predicted,expected", [([1, 1, 0, 1], [1, 0, 0, 1], 0.75)]
)
def test_base_accuracy(accuracy, y_observed, y_predicted, expected):
    actual = accuracy(y_observed=y_observed, y_predicted=y_predicted)
    assert expected == actual


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y_observed,y_predicted,sample_weights,expected",
    [([1, 1, 0, 1], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.75)],
)
def test_weighted_accuracy(accuracy, y_observed, y_predicted, sample_weights, expected):
    actual = accuracy(
        y_observed=y_observed, y_predicted=y_predicted, sample_weights=sample_weights
    )
    assert expected == actual
