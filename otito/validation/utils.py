import functools


def argument_validator(validator_model):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            args = validator_model(*args, **kwargs)
            return func(**args.dict())

        return inner_wrapper

    return outer_wrapper
