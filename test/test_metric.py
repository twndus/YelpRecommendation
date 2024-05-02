import numpy as np
from math import isclose

from ..metric import (
    precision_at_k, recall_at_k, map_at_k
)


def test_precision_at_k():
    actual: np.ndarray = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    predicted: np.ndarray = np.array([[1,6,7,11,12], [6,7,14,16,20]])

    assert isclose(precision_at_k(actual, predicted, 1), 1)
    assert isclose(precision_at_k(actual, predicted, 2), 0.75)
    assert isclose(precision_at_k(actual, predicted, 3), 0.5)
    assert isclose(precision_at_k(actual, predicted, 4), 0.375)
    assert isclose(precision_at_k(actual, predicted, 5), 0.3)

def test_recall_at_k():
    actual: np.ndarray = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    predicted: np.ndarray = np.array([[1,6,7,11,12], [6,7,14,16,20]])

    assert isclose(recall_at_k(actual, predicted, 1), 0.2)
    assert isclose(recall_at_k(actual, predicted, 2), 0.3)
    assert isclose(recall_at_k(actual, predicted, 3), 0.3)
    assert isclose(recall_at_k(actual, predicted, 4), 0.3)
    assert isclose(recall_at_k(actual, predicted, 5), 0.3)

def test_map_at_k():
    actual: np.ndarray = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    predicted: np.ndarray = np.array([[1,6,7,11,12], [6,7,14,16,20]])

    assert isclose(map_at_k(actual, predicted, 1), 0.2) # (1/5 + 1/5) / 2
    assert isclose(map_at_k(actual, predicted, 2), 0.3) # (1/5 + 2/5) / 2
    assert isclose(map_at_k(actual, predicted, 3), 0.3) # (1/5 + 2/5) / 2
    assert isclose(map_at_k(actual, predicted, 4), 0.3) # (1/5 + 2/5) / 2
    assert isclose(map_at_k(actual, predicted, 5), 0.3) # (1/5 + 2/5) / 2
