import pytest


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y,y_pred,expected",
    [
        ([0, 0, 0, 0], [1, 1, 1, 1], 0),
        ([1, 1, 1, 1], [0, 0, 0, 0], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1),
        ([1, 1, 0, 1], [1, 0, 0, 1], 0.75),
    ],
)
def test_base_accuracy(accuracy, y, y_pred, expected):
    actual = accuracy(y=y, y_pred=y_pred)
    assert expected == actual


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y,y_pred,sample_weights,expected",
    [
        ([1, 1, 0, 1], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.75),
        ([0, 0, 0, 0], [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], 0.75),
    ],
)
def test_weighted_accuracy(accuracy, y, y_pred, sample_weights, expected):
    actual = accuracy(y=y, y_pred=y_pred, sample_weights=sample_weights)
    assert expected == actual
