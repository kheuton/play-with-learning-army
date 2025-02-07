import torch
import copy

class L2SPLoss(torch.nn.Module):
    def __init__(self, alpha, bb_loc, beta, criterion=torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.bb_loc = bb_loc
        self.beta = beta
        self.criterion = criterion
        self.D = len(self.bb_loc)

    def forward(self, logits, labels, weights, params):

        nll = self.criterion(logits, labels)*weights
        nll = nll.mean()
        bb_log_prob = (self.alpha/2) * ((params[:self.D] - self.bb_loc)**2).sum()
        clf_log_prob = (self.beta/2) * (params[self.D:]**2).sum()
        
        loss = nll + bb_log_prob + clf_log_prob
        return loss, nll, bb_log_prob, clf_log_prob
    
class WeightedBCELoss(torch.nn.Module,):
    def __init__(self, criterion=torch.nn.BCELoss(reduction='none')):
        super().__init__()
        self.criterion = criterion

    def forward(self, logits, labels, weights, params):
        nll = self.criterion(logits, labels)*weights
        nll = nll.sum()
        return nll

def initialize_loss(hyper_config, embedder):

    if hyper_config['loss'] == 'l2sp':
        params = torch.nn.utils.parameters_to_vector(embedder.model.parameters()).detach()
        bb_loc = params
        return L2SPLoss(hyper_config['alpha'], bb_loc, hyper_config['beta'], criterion=torch.nn.BCELoss(reduction='none'))
    else:
        return WeightedBCELoss()

def initialize_optimizer(hyper_config, domain_model_dict, embed_func):
    params = []

    if hyper_config['finetune']:
        params += [param for param in embed_func.model.parameters()]
    for model in domain_model_dict.values():
        params += [param for param in model.parameters()]
    
    return torch.optim.Adam(params,
                            lr=hyper_config['learning_rate'], weight_decay=hyper_config['opt_weight_decay'])