import pytest
import tensorflow as tf

from otito.metrics.utils import load_metric


class TestBinaryAccuracy:
    """
    Class to test the tensorflow binary accuracy metric
    """

    @pytest.fixture
    def metric(self):
        return load_metric(
            metric="BinaryAccuracy", package="tensorflow", validate_input=True
        )

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        "y_observed,y_predicted,expected",
        [
            (tf.constant([0, 0, 0, 0]), tf.constant([1, 1, 1, 1]), 0),
            (tf.constant([1, 1, 1, 1]), tf.constant([0, 0, 0, 0]), 0),
            (tf.constant([1, 1, 1, 1]), tf.constant([1, 1, 1, 1]), 1),
            (tf.constant([0, 0, 0, 0]), tf.constant([0, 0, 0, 0]), 1),
            (tf.constant([1, 1, 0, 1]), tf.constant([1, 0, 0, 1]), 0.75),
        ],
    )
    def test_base_accuracy(self, metric, y_observed, y_predicted, expected):
        actual = metric(y_observed=y_observed, y_predicted=y_predicted)
        assert expected == pytest.approx(actual)

    @pytest.mark.usefixtures("metric")
    @pytest.mark.parametrize(
        "y_observed,y_predicted,sample_weights,expected",
        [
            (
                tf.constant([1, 1, 0, 1]),
                tf.constant([1, 0, 0, 1]),
                tf.constant([0.25, 0.25, 0.25, 0.25]),
                0.75,
            ),
            (
                tf.constant([0, 0, 0, 0]),
                tf.constant([1, 1, 1, 1]),
                tf.constant([0.25, 0.25, 0.25, 0.25]),
                0.0,
            ),
        ],
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


# @pytest.mark.usefixtures("metric")
# @pytest.mark.parametrize(
#     "y_observed,y_predicted",
#     [
#         (tf.constant([1, 1, 1, 1]), tf.constant([1, 1, 1])),
#         (tf.constant([1, 1, 1]), tf.constant([1, 1, 1, 1])),
#     ],
# )
# def test_labels_must_be_same_shape(self, metric, y_observed, y_predicted):
#     expected_msg = (
#         f"1 validation error for tensorflow:BinaryAccuracyModel\ny_predicted\n  "
#         f"Shape of inputs mismatched: {len(y_observed)} "
#         f"(observed) != {len(y_predicted)} (predicted) (type=value_error)"
#     )
#     with pytest.raises(ValueError) as e:
#         metric.validator(y_observed=y_observed, y_predicted=y_predicted)
#
#     assert expected_msg == str(e.value)
#
# @pytest.mark.usefixtures("metric")
# @pytest.mark.parametrize(
#     "y_observed,y_predicted",
#     [
#         (tf.constant([1, 2, 3]), tf.constant([0, 1, 1])),
#         (tf.constant([2, 0, 0]), tf.constant([0, 1, 1])),
#         (
#             tf.constant([0, 1, 0]),
#             tf.constant([0.999, 1, 1]),
#         ),
#     ],
# )
# def test_labels_must_be_binary(self, metric, y_observed, y_predicted):
#     found_classes = list(
#         set(np.concatenate((y_observed.numpy(), y_predicted.numpy())))
#     )
#     expected_msg = (
#         f"1 validation error for tensorflow:BinaryAccuracyModel\ny_predicted\n  "
#         f"Input is not binary: '{len(found_classes)}' class labels "
#         "found (type=value_error)"
#     )
#     with pytest.raises(ValueError) as e:
#         metric.validator(y_observed=y_observed, y_predicted=y_predicted)
#
#     assert expected_msg == str(e.value)
#
# @pytest.mark.usefixtures("metric")
# @pytest.mark.parametrize(
#     "y_observed,y_predicted,sample_weights",
#     [
#         (
#             tf.constant([1, 1, 0]),
#             tf.constant([0, 1, 1]),
#             tf.constant([0.5, 0.5]),
#         ),
#         (
#             tf.constant([1, 0, 0]),
#             tf.constant([0, 1, 1]),
#             tf.constant([0.25, 0.25, 0.25, 0.25]),
#         ),
#     ],
# )
# def test_sample_weights_must_be_same_len(
#     self, metric, y_observed, y_predicted, sample_weights
# ):
#     expected_msg = (
#         f"1 validation error for tensorflow:BinaryAccuracyModel\nsample_weights\n  "
#         "'sample_weights' is not the same length as input. "
#         f"Lengths (sample_weights:{len(sample_weights)}, "
#         f"input:{len(y_observed)}) (type=value_error)"
#     )
#     with pytest.raises(ValueError) as e:
#         metric.validator(
#             y_observed=y_observed,
#             y_predicted=y_predicted,
#             sample_weights=sample_weights,
#         )
#
#     assert expected_msg == str(e.value)
#
# @pytest.mark.usefixtures("metric")
# @pytest.mark.parametrize(
#     "y_observed,y_predicted,sample_weights",
#     [
#         (
#             tf.constant([1, 1, 0]),
#             tf.constant([0, 1, 1]),
#             tf.constant([0.5, 0.5, 0.3]),
#         ),
#         (
#             tf.constant([1, 0, 0]),
#             tf.constant([0, 1, 1]),
#             tf.constant([0.1, 0.1, 0.1]),
#         ),
#     ],
# )
# def test_weights_must_sum_to_one(
#     self, metric, y_observed, y_predicted, sample_weights
# ):
#     expected_msg = (
#         f"1 validation error for tensorflow:BinaryAccuracyModel\nsample_weights\n  "
#         "'sample_weights' do not sum to one. "
#         f"Sum of `sample_weights`:{sample_weights.sum()} (type=value_error)"
#     )
#     with pytest.raises(ValueError) as e:
#         metric.validator(
#             y_observed=y_observed,
#             y_predicted=y_predicted,
#             sample_weights=sample_weights,
#         )
#
#     assert expected_msg == str(e.value)
