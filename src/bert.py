from transformers import BertModel, BertTokenizer
from sklearn.base import TransformerMixin, BaseEstimator
import torch

def initialize_bert(train_dataset, hyper_config):
    return BertEmbedder(finetune=hyper_config['finetune'], device=hyper_config['device'])

def initialize_frozen_bert(train_dataset, hyper_config):
    return FrozenBertEmbedder(finetune=hyper_config['finetune'], device=hyper_config['device'])

class BertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', max_length=512, device=None, finetune=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model.to(self.device)
        self.embedding_size = 768+1

    def forward(self, X):
        
        model_output = self.model(**X)

        # We typically use the embeddings from the [CLS] token (first token)
        embeddings = model_output.last_hidden_state[:, 0, :]
        return embeddings
    
    def preprocess_data(self, train_dataset, hyper_config):

        x, y, problem_id, student_id = train_dataset
        
        encoded_input = self.tokenizer(
            x, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        ).to(self.device)

        return encoded_input, torch.tensor(y).to(self.device), problem_id.to(self.device), student_id.to(self.device)
    
class FrozenBertEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased', max_length=512, device=None, finetune=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model.to(self.device)
        self.embedding_size = 768+1

    def forward(self, X):

        return X
    
    def preprocess_data(self, train_dataset, hyper_config):

        x, y, problem_id, student_id = train_dataset
        
        encoded_input = self.tokenizer(
            x, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        ).to(self.device)

        embedded_input = self.model(**encoded_input).last_hidden_state[:, 0, :]

        return embedded_input, torch.tensor(y).to(self.device), problem_id.to(self.device), student_id.to(self.device)