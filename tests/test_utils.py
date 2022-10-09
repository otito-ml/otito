def convert_test_data(values, columns, target_type, **kwargs):
    return [
        values[0],
        [
            tuple(
                target_type(element, **kwargs) if idx in columns else element
                for idx, element in enumerate(sample)
            )
            for sample in values[1]
        ],
    ]
