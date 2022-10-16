import pytest
import torch as pt
import numpy as np

from otito.metrics.utils import load_metric

from tests.test_utils import get_cases
from tests.metrics.classification.binary.resources import test_data as td


class TestBinaryAccuracy:
    """
    Class to test the pytorch binary accuracy metric
    """

    target_type = pt.tensor

    @pytest.fixture
    def metric(self):
        return load_metric(
            metric="BinaryAccuracy",
            package="pytorch",
            validate_input=True,
            stateful=False,
        )

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_cases(
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
        *get_cases(
            data_module=td,
            data_name="weighted_accuracy_data",
            columns=[0, 1, 2],
            target_type=target_type,
        )
    )
    def test_weighted_accuracy(
        self, metric, y_observed, y_predicted, sample_weight, expected
    ):
        actual = metric(
            y_observed=y_observed,
            y_predicted=y_predicted,
            sample_weight=sample_weight,
        )
        assert expected == pytest.approx(actual)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_cases(
            data_module=td,
            data_name="labels_must_be_same_shape_data",
            columns=[0, 1],
            target_type=target_type,
        )
    )
    def test_labels_must_be_same_shape(self, metric, y_observed, y_predicted):
        expected_msg = (
            f"1 validation error for pytorch:BinaryAccuracyModel\ny_predicted\n  "
            f"Shape of inputs mismatched: {len(y_observed)} "
            f"(observed) != {len(y_predicted)} (predicted) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_cases(
            data_module=td,
            data_name="labels_must_be_binary_data",
            columns=[0, 1],
            target_type=target_type,
        )
    )
    def test_labels_must_be_binary(self, metric, y_observed, y_predicted):
        found_classes = list(
            set(np.concatenate((y_observed.numpy(), y_predicted.numpy())))
        )
        expected_msg = (
            f"1 validation error for pytorch:BinaryAccuracyModel\ny_predicted\n  "
            f"Input is not binary: '{len(found_classes)}' class labels "
            "found (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(y_observed=y_observed, y_predicted=y_predicted)

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_cases(
            data_module=td,
            data_name="sample_weight_must_be_same_len_data",
            columns=[0, 1, 2],
            target_type=target_type,
        )
    )
    def test_sample_weight_must_be_same_len(
        self, metric, y_observed, y_predicted, sample_weight
    ):
        expected_msg = (
            f"1 validation error for pytorch:BinaryAccuracyModel\nsample_weight\n  "
            "'sample_weight' is not the same length as input. "
            f"Lengths (sample_weight:{len(sample_weight)}, "
            f"input:{len(y_observed)}) (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weight=sample_weight,
            )

        assert expected_msg == str(e.value)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        *get_cases(
            data_module=td,
            data_name="weights_must_sum_to_one_data",
            columns=[0, 1, 2],
            target_type=target_type,
        )
    )
    def test_weights_must_sum_to_one(
        self, metric, y_observed, y_predicted, sample_weight
    ):
        expected_msg = (
            f"1 validation error for pytorch:BinaryAccuracyModel\nsample_weight\n  "
            "'sample_weight' do not sum to one. "
            f"Sum of `sample_weight`:{sample_weight.sum()} (type=value_error)"
        )
        with pytest.raises(ValueError) as e:
            metric.validator(
                y_observed=y_observed,
                y_predicted=y_predicted,
                sample_weight=sample_weight,
            )

        assert expected_msg == str(e.value)
