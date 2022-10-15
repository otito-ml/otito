base_accuracy_data = {
    "fields": "y_observed,y_predicted,expected",
    "values": [
        ([0, 0, 0, 0], [1, 1, 1, 1], 0),
        ([1, 1, 1, 1], [0, 0, 0, 0], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1),
        ([1, 1, 0, 1], [1, 0, 0, 1], 0.75),
    ],
}


weighted_accuracy_data = {
    "fields": "y_observed,y_predicted,sample_weight,expected",
    "values": [
        ([1, 1, 0, 1], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.75),
        ([0, 0, 0, 0], [1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], 0.0),
    ],
}


labels_must_be_same_shape_data = {
    "fields": "y_observed,y_predicted",
    "values": [
        ([1, 1, 1, 1], [1, 1, 1]),
        ([1, 1, 1], [1, 1, 1, 1]),
    ],
}

labels_must_be_binary_data = {
    "fields": "y_observed,y_predicted",
    "values": [
        ([1, 2, 3], [0, 1, 1]),
        ([0, 0, 0], [0, -1, 1]),
        ([0, 1, 0], [0.999999, 1, 1]),
    ],
}

sample_weight_must_be_same_len_data = {
    "fields": "y_observed,y_predicted,sample_weight",
    "values": [
        ([1, 1, 0], [0, 1, 1], [0.5, 0.5]),
        ([1, 0, 0], [0, 1, 1], [0.25, 0.25, 0.25, 0.25]),
    ],
}


weights_must_sum_to_one_data = {
    "fields": "y_observed,y_predicted,sample_weight",
    "values": [
        ([1, 1, 0], [0, 1, 1], [0.5, 0.5, 0.3]),
        ([1, 0, 0], [0, 1, 1], [0.1, 0.1, 0.1]),
    ],
}
