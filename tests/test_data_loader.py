import pytest
import pandas as pd
from src.data_loader import read_xlsx

class MockConfig:
    def __init__(self, M):
        self.M = M  # Defines the number of output columns

@pytest.fixture
def sample_xlsx(tmp_path):
    """Create a temporary Excel file for testing."""
    file_path = tmp_path / "test_data.xlsx"
    df = pd.DataFrame({"text": ["hello", "world"], "label_1": [1, 0], "label_2": [0, 1]})
    df.to_excel(file_path, index=False)
    return str(file_path)

def test_read_xlsx(sample_xlsx):
    config = MockConfig(M=2)  # Expecting y array with 2 columns
    X, y = read_xlsx(sample_xlsx, config)

    assert len(X) == 2
    assert X[0] == "hello"
    assert X[1] == "world"
    assert y.shape == (2, 2)
