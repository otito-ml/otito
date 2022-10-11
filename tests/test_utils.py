def get_cases(data_module, data_name, columns, target_type, **kwargs):
    values = list(getattr(data_module, data_name).values())
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
