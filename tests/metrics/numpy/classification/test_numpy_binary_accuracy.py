import pytest
import numpy as np

from otito.metrics.numpy import BinaryAccuracy


class TestAccuracy:
    """
    Class to test Numpy Accuracy Metric Function
    """

    @pytest.fixture
    def accuracy(self):
        return BinaryAccuracy(package="numpy", validate_input=True)

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

    @pytest.mark.usefixtures("accuracy")
    @pytest.mark.parametrize(
        "y_observed,y_predicted",
        [
            (np.array([1, 1, 1, 1], dtype=float), np.array([1, 1, 1], dtype=float)),
            (np.array([1, 1, 1], dtype=float), np.array([1, 1, 1, 1], dtype=float)),
        ],
    )
    def test_labels_must_be_same_shape(self, accuracy, y_observed, y_predicted):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\ny_predicted\n  "
            f"Shape of inputs mismatched: {y_observed.shape[0]} "
            f"(observed) != {y_predicted.shape[0]} (predicted) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            accuracy.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("accuracy")
    @pytest.mark.parametrize(
        "y_observed,y_predicted",
        [
            (np.array([1, 2, 3], dtype=float), np.array([0, 1, 1], dtype=float)),
            (np.array([0, 0, 0], dtype=float), np.array([0, -1, 1], dtype=float)),
            (
                np.array([0, 1, 0], dtype=float),
                np.array([0.999999, 1, 1], dtype=float),
            ),
        ],
    )
    def test_labels_must_be_binary(self, accuracy, y_observed, y_predicted):
        found_classes = sorted(list(set(np.concatenate((y_observed, y_predicted)))))
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\ny_predicted\n  "
            f"Input is not binary: '{len(found_classes)}' class labels "
            "found (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            accuracy.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("accuracy")
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
    def test_sample_weights_must_be_same_len(
        self, accuracy, y_observed, y_predicted, sample_weights
    ):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\nsample_weights\n  "
            "'sample_weights' is not the same length as input. "
            f"Lengths (sample_weights:{len(sample_weights)}, "
            f"input:{len(y_observed)}) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            accuracy.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("accuracy")
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
    def test_weights_must_sum_to_one(
        self, accuracy, y_observed, y_predicted, sample_weights
    ):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\nsample_weights\n  "
            "'sample_weights' do not sum to one. "
            f"Sum of `sample_weights`:{sample_weights.sum()} (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            accuracy.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)
