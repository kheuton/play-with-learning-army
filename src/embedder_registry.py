from bag_of_words import initialize_bow
from bert import initialize_bert
import torch

REGISERTED_EMBEDDERS = {'bow':  initialize_bow,
                        'bert': initialize_bert}



def initialize_embedding(hyper_config, train_dataset):
    embedder = REGISERTED_EMBEDDERS[hyper_config['embedder']]
    return embedder(train_dataset)


def identity_criteria_embedder(criteria):
    return criteria

REGISTERED_CRITERIA_EMBEDDERS = {'identity': identity_criteria_embedder}

def initialize_criteria_embedding(hyper_config):
    embedder = REGISTERED_CRITERIA_EMBEDDERS[hyper_config['criteria_embedder']]
    return embedder

def concatenate_combiner(embedding, criteria):
    return torch.cat((embedding, criteria), 1)

REGISTERED_COMBINERS = {'concatenate': concatenate_combiner}

def initialize_combiner(hyper_config):
    combiner = REGISTERED_COMBINERS[hyper_config['combiner']]
    return combiner
