def call_metric(func, validate=True, *args, **kwargs):
    if validate:
        return func(*args, **kwargs)
    else:
        return func.__wrapped__(*args, **kwargs)
