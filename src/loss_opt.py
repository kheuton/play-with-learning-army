import torch

def initialize_loss(hyper_config):
    return torch.nn.BCELoss()

def initialize_optimizer(hyper_config, domain_model_dict, embed_func):
    params = []
    for model in domain_model_dict.values():
        params += [param for param in model.parameters()]
    if hyper_config['finetune']:
        params += [param for param in embed_func.model.parameters()]
    return torch.optim.Adam( params,
                            lr=hyper_config['learning_rate'], weight_decay=hyper_config['weight_decay'])