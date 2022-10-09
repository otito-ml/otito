import pytest
import numpy as np

from otito.metrics.utils import load_metric

from tests.test_utils import get_test_data
from tests.metrics.classification.binary.resources import test_data as td


class TestBinaryAccuracy:
    """
    Class to test numpy binary accuracy metric
    """

    target_type = np.array

    @pytest.fixture
    def metric(self):
        return load_metric(
            metric="BinaryAccuracy", package="numpy", validate_input=True
        )

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="base_accuracy_data",
            columns=[0, 1],
            target_type=target_type,
        )
    )
    def test_base_accuracy(self, metric, y_observed, y_predicted, expected):
        actual = metric(y_observed=y_observed, y_predicted=y_predicted)
        assert expected == pytest.approx(actual)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="weighted_accuracy_data",
            columns=[0, 1, 2],
            target_type=target_type,
        )
    )
    def test_weighted_accuracy(
        self, metric, y_observed, y_predicted, sample_weights, expected
    ):
        actual = metric(
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weights=sample_weights,
        )
        assert expected == pytest.approx(actual)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="labels_must_be_same_shape_data",
            columns=[0, 1],
            target_type=target_type,
            dtype=float,
        )
    )
    def test_labels_must_be_same_shape(self, metric, y_observed, y_predicted):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\ny_predicted\n  "
            f"Shape of inputs mismatched: {y_observed.shape[0]} "
            f"(observed) != {y_predicted.shape[0]} (predicted) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="labels_must_be_binary_data",
            columns=[0, 1],
            target_type=target_type,
            dtype=float,
        )
    )
    def test_labels_must_be_binary(self, metric, y_observed, y_predicted):
        found_classes = sorted(list(set(np.concatenate((y_observed, y_predicted)))))
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\ny_predicted\n  "
            f"Input is not binary: '{len(found_classes)}' class labels "
            "found (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="sample_weights_must_be_same_len_data",
            columns=[0, 1, 2],
            target_type=target_type,
            dtype=float,
        )
    )
    def test_sample_weights_must_be_same_len(
        self, metric, y_observed, y_predicted, sample_weights
    ):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\nsample_weights\n  "
            "'sample_weights' is not the same length as input. "
            f"Lengths (sample_weights:{len(sample_weights)}, "
            f"input:{len(y_observed)}) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_test_data(
            data_module=td,
            data_name="weights_must_sum_to_one_data",
            columns=[0, 1, 2],
            target_type=target_type,
            dtype=float,
        )
    )
    def test_weights_must_sum_to_one(
        self, metric, y_observed, y_predicted, sample_weights
    ):
        expected_msg = (
            f"1 validation error for numpy:BinaryAccuracyModel\nsample_weights\n  "
            "'sample_weights' do not sum to one. "
            f"Sum of `sample_weights`:{sample_weights.sum()} (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weights=sample_weights,
            )

        assert expected_msg == str(e.value)
