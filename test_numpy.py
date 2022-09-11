from sklearn.metrics import make_scorer
from otito.metrics import load_metric
import numpy as np
import timeit

from sklearn.svm import LinearSVC


def basic_accuracy():

    accuracy_score = load_metric(metric="Accuracy", package="numpy", parse_input=True)
    start_time = timeit.default_timer()
    result = accuracy_score(
        np.array([1, 1, 0]),
        np.array([1, 0, 0]),
    )
    print(f"Otito (Numpy): {timeit.default_timer()-start_time}: Result: {result}")


############################################################
# SKLEARN SCORER
############################################################


def sklearn_scorer():
    accuracy_score = load_metric(metric="Accuracy", package="numpy", parse_input=True)
    otito_accuracy_scorer = make_scorer(accuracy_score)

    x_train = [[1] * 3] * 5 + [[0] * 3] * 5
    y_train = [1] * 5 + [0] * 5

    x_test = [[1] * 3] * 5 + [[0] * 3] * 5
    y_test = [1] * 2 + [0] * 2 + [1] * 2 + [0] * 2 + [1] * 2

    clf = LinearSVC(C=1)
    clf.fit(x_train, y_train)

    print(otito_accuracy_scorer(clf, x_test, y_test))


if __name__ == "__main__":
    basic_accuracy()
    sklearn_scorer()
