import numpy as np
from math import isclose

from ..metric import precision_at_k

def test_precision_at_k():
    actual: np.ndarray = np.array([[1,2,3,4,5], [6,7,8,9,10]])
    predicted: np.ndarray = np.array([[1,6,7,11,12], [6,7,14,16,20]])

    assert isclose(precision_at_k(actual, predicted, 2), 0.75)
    assert isclose(precision_at_k(actual, predicted, 5), 0.3)
