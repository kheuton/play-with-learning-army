import torch
import copy

class L2SPLoss(torch.nn.Module):
    def __init__(self, alpha, bb_loc, beta, model, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.bb_loc = bb_loc
        self.beta = beta
        self.criterion = criterion
        self.D = len(self.bb_loc)
        self.model = model

    def forward(self, labels, logits):
        params = torch.cat([param for param in self.model.parameters()])
        nll = self.criterion(logits, labels)
        bb_log_prob = (self.alpha/2) * ((params[:self.D] - self.bb_loc.to(params.device))**2).sum()
        clf_log_prob = (self.beta/2) * (params[self.D:]**2).sum()
        return {'bb_log_prob': bb_log_prob, 'clf_log_prob': clf_log_prob, 'nll': nll, 'loss': nll + bb_log_prob + clf_log_prob}

def initialize_loss(hyper_config, embedder):

    if hyper_config['loss'] == 'l2sp':
        params = torch.nn.utils.parameters_to_vector(embedder.model.parameters())
        bb_loc = copy.deepcopy(params.detach())
        return L2SPLoss(hyper_config['alpha'], bb_loc, hyper_config['beta'], embedder, criterion=torch.nn.BCELoss())
    else:
        return torch.nn.BCELoss()

def initialize_optimizer(hyper_config, domain_model_dict, embed_func):
    params = []

    if hyper_config['finetune']:
        params += torch.nn.utils.parameters_to_vector(embed_func.model.parameters())
    for model in domain_model_dict.values():
        params += torch.nn.utils.parameters_to_vector(model.parameters())
    
    return torch.optim.Adam( params,
                            lr=hyper_config['learning_rate'], weight_decay=hyper_config['weight_decay'])