import numpy as np


def _base_accuracy(y_observed, y_predicted):
    return np.average(np.array(y_observed) == np.array(y_predicted))
