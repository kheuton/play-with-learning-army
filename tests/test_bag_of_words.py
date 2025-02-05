import pytest
from src.bag_of_words import embed_bow

@pytest.fixture
def sample_vocab():
    return {"hello": 0, "world": 1, "python": 2}

def test_embed_bow_with_vocab(sample_vocab):
    text = "hello world hello"
    bow = embed_bow(text, sample_vocab)

    assert bow.shape == (len(sample_vocab),)
    assert bow[0] == 2  # "hello" appears twice
    assert bow[1] == 1  # "world" appears once
    assert bow[2] == 0  # "python" is not in the text