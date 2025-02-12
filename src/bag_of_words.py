import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

def initialize_bow(train_dataset, hyper_config):
    return BagOfWordsEmbedder(train_dataset)

class BagOfWordsEmbedder(torch.nn.Module):
    def __init__(self, train_dataset):
        self.vectorizer = CountVectorizer()
        x,y,problem_id,student_id = train_dataset
        self.vectorizer.fit_transform(x)
        vocab = self.vectorizer.vocabulary_
        vocab_size = len(vocab)
        self.embedding_size = vocab_size + 1
        self.device = 'cpu'

    def preprocess_data(self, dataset, hyper_config):
        x, y, problem_id, student_id = dataset
        return torch.tensor(self.vectorizer.transform([x]).todense(), dtype=torch.float32), torch.tensor(y).to(self.device), problem_id.to(self.device), student_id.to(self.device)

    def forward(self, texts):
        return texts
