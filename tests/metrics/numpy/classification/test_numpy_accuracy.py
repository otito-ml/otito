import pytest
import numpy as np

from otito.metrics.numpy.validation.classification import BaseAccuracyValidator
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

    @pytest.mark.parametrize(
        "y_observed,y_predicted",
        [
            (np.array([1, 1, 1, 1], dtype=float), np.array([1, 1, 1], dtype=float)),
            (np.array([1, 1, 1], dtype=float), np.array([1, 1, 1, 1], dtype=float)),
        ],
    )
    def test_input_must_be_same_shape(self, y_observed, y_predicted):
        expected_msg = (
            f"1 validation error for BaseAccuracyValidator\n__root__\n  "
            f"Shape of inputs mismatched: {len(y_observed)} "
            f"(observed) != {len(y_predicted)} (predicted) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            BaseAccuracyValidator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.parametrize(
        "y_observed,y_predicted",
        [
            (np.array([1, 2, 3], dtype=float), np.array([0, 1, 1], dtype=float)),
            (np.array([None, 0, 0], dtype=float), np.array([0, 1, 1], dtype=float)),
            (
                np.array([0, 1, 0], dtype=float),
                np.array([0.999999999999999, 1, 1], dtype=float),
            ),
        ],
    )
    def test_input_must_be_binary(self, y_observed, y_predicted):
        found_classes = list(set(np.concatenate((y_observed, y_predicted))))
        expected_msg = (
            f"1 validation error for BaseAccuracyValidator\n__root__\n  "
            f"Input is not binary: '{len(found_classes)}' "
            f"classes found. Found classes: `{found_classes}` (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            BaseAccuracyValidator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.parametrize(
        "y_observed,y_predicted,sample_weights",
        [
            (
                np.array([1, 1, 0], dtype=float),
                np.array([0, 1, 1], dtype=float),
                np.array([0.5, 0.5], dtype=float),
            ),
            (
                np.array([1, 0, 0], dtype=float),
                np.array([0, 1, 1], dtype=float),
                np.array([0.25, 0.25, 0.25, 0.25], dtype=float),
            ),
        ],
    )
    def test_weights_must_be_same_length_as_samples(
        self, y_observed, y_predicted, sample_weights
    ):
        expected_msg = (
            f"1 validation error for BaseAccuracyValidator\n__root__\n  "
            "'sample_weights' is not the same length as input. "
            f"Lengths (sample_weights:{len(sample_weights)}, "
            f"input:{len(y_observed)}) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            BaseAccuracyValidator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)

    @pytest.mark.parametrize(
        "y_observed,y_predicted,sample_weights",
        [
            (
                np.array([1, 1, 0], dtype=float),
                np.array([0, 1, 1], dtype=float),
                np.array([0.5, 0.5, 0.3], dtype=float),
            ),
            (
                np.array([1, 0, 0], dtype=float),
                np.array([0, 1, 1], dtype=float),
                np.array([0.1, 0.1, 0.1], dtype=float),
            ),
        ],
    )
    def test_weights_must_sum_to_one(self, y_observed, y_predicted, sample_weights):
        expected_msg = (
            f"1 validation error for BaseAccuracyValidator\n__root__\n  "
            "'sample_weights' do not sum to one. "
            f"Sum of `sample_weights`:{sample_weights.sum()} (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            BaseAccuracyValidator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)
