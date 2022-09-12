import pytest


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y,y_predicted,expected",
    [
        ([0, 0, 0, 0], [1, 1, 1, 1], 0),
        ([1, 1, 1, 1], [0, 0, 0, 0], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1),
        ([1, 1, 0, 1], [1, 0, 0, 1], 0.75),
    ],
)
def test_base_accuracy(accuracy, y, y_predicted, expected):
    actual = accuracy(y_observed=y, y_predicted=y_predicted)
    assert expected == actual


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y,y_predicted,sample_weights,expected",
    [
        ([1, 1, 0, 1], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.75),
        ([0, 0, 0, 0], [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], 0.0),
    ],
)
def test_weighted_accuracy(accuracy, y, y_predicted, sample_weights, expected):
    actual = accuracy(
        y_observed=y, y_predicted=y_predicted, sample_weights=sample_weights
    )
    assert expected == actual
