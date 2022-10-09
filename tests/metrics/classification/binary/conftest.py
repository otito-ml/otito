from resources import test_data as td


def get_test_data(data):
    return getattr(td, data)
