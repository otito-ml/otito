import numpy as np


def accuracy(y_observed, y_predicted):
    return np.average(
        np.array(y_observed) == np.array(y_predicted)
    )
