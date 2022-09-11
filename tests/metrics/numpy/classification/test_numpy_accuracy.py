import pytest


@pytest.mark.usefixtures("accuracy")
@pytest.mark.parametrize(
    "y_observed,y_predicted,expected", [([1, 1, 0, 1], [1, 0, 0, 1], 0.75)]
)
def test_base_accuracy(accuracy, y_observed, y_predicted, expected):
    actual = accuracy(y_observed=y_observed, y_predicted=y_predicted)
    assert expected == actual
