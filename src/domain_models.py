import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.linear(x))
    

def create_domain_dict(num_domains, embedding_size, device):
    domain_model_dict = {}
    for d in range(num_domains):
        domain_model_dict[d] = LogisticRegression(embedding_size).to(device)
    return domain_model_dict

def initialize_domain_models(embedding_size, device, problem_config):
    domain_model_dict = create_domain_dict(problem_config["num_domains"], embedding_size,device)
    return domain_model_dict