import pytest
from otito.metrics.numpy.classification.accuracy import Accuracy


@pytest.fixture
def accuracy():
    return Accuracy()
