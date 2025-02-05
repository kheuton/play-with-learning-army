import numpy as np

def embed_bow(text, vocab):
    bow = np.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            bow[vocab[word]] += 1
    return bow