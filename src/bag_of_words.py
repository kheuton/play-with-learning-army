import numpy as np
import torch


class BagOfWords(torch.nn.Module):
    def __init__(self, train_dataset, hyper_config):
        super(BagOfWords, self).__init__()
        self.vocab = self.build_vocab(train_dataset)

    def build_vocab(self, train_dataset):
        vocab = set()
        for x in train_dataset:
            vocab.update(x)
        return vocab
    
    def forward(self, x):
        # x is a list of words
        # return the bag-of-words embedding of x
        embedding = torch.zeros(len(self.vocab))
        for word in x:
            embedding[self.vocab.index(word)] += 1
        return embedding
    