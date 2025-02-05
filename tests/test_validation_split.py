import pytest
import numpy as np
from src.validation_split import data_split

@pytest.fixture
def sample_data():
    X = np.array(["text1", "text2", "text3", "text4", "text5"])
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    return X, y

def test_data_split(sample_data):
    X, y = sample_data
    val_frac, test_frac = 0.2, 0.2
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_split(X, y, val_frac, test_frac)

    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)
    assert len(X_val) == int(len(X) * val_frac)
    assert len(X_test) == int(len(X) * test_frac)
