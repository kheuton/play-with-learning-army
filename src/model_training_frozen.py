from data_loader import load_datasets, to_device
from embedder_registry import initialize_embedding, initialize_criteria_embedding, initialize_combiner
from domain_models import initialize_domain_models
from loss_opt import initialize_loss, initialize_optimizer
import torch
import yaml
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
import transformers

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_experiment(hyper_config, problem_config, use_wandb=False):
    """
    Run a single experiment
    """

    set_seed(hyper_config['seed'])

    if 'use_wandb' in hyper_config:
        use_wandb = hyper_config['use_wandb']
    else:
        use_wandb = True
    
    # Load data
    train_datasets, val_datasets = load_datasets(hyper_config,
                                                 train=True, val=True, test=False)
    
    for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        if use_wandb:
            wandb.init(project=hyper_config['wandb_project'], config=hyper_config, name=hyper_config['experiment_name'].format(fold=fold))
    
        embedder = initialize_embedding(hyper_config, train_dataset)
        criteria_embedder = initialize_criteria_embedding(hyper_config)
        criteria_combiner = initialize_combiner(hyper_config)

        domain_model_dict = initialize_domain_models(embedder.embedding_size,  embedder.device, problem_config)
        loss_func = initialize_loss(hyper_config, embedder)
        optimizer = initialize_optimizer(hyper_config, domain_model_dict, embedder)

        model = train_model(fold, train_dataset, val_dataset,
                            embedder, criteria_embedder, criteria_combiner,
                            domain_model_dict,
                            loss_func,
                            optimizer,
                            hyper_config, problem_config, use_wandb)
    
    return

def evaluate_model(dataloader, hyper_config, embed_func, criteria_embed_func, criteria_combiner, domain_model_dict, loss_func, problem_config, num_domains):
    total_loss, total_nll, total_bb_log_prob, total_clf_log_prob, total_unweighted_nll = 0, 0, 0, 0, 0
    all_preds, all_labels, all_weights = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            X_batch, y_batch, p_batch, s_batch = batch
            if isinstance(X_batch, dict):
                batch_length = len(next(iter(X_batch.values())))
            else:
                batch_length = len(X_batch)
                
            for i in range(batch_length):
                if isinstance(X_batch, dict):
                    x = {k: v[i].unsqueeze(0) for k, v in X_batch.items()}
                else:
                    x = X_batch[i].unsqueeze(0)
                y = y_batch[i]
                p = p_batch[i]
                s = s_batch[i]

                x_embed = embed_func.forward(x)
                
                criteria_counter = 0
                for d in range(num_domains):
                    num_criteria = problem_config['problems'][p]['domains'][d]["num_criteria"]
                    for c in range(num_criteria):
                        c_embed = criteria_embed_func(torch.tensor([[c]]).to(embed_func.device))
                        final_representation = criteria_combiner(x_embed, c_embed)
                        y_pred = domain_model_dict[d](final_representation)
                        weight = 1/num_criteria

                        all_preds.append(y_pred.detach().cpu().numpy())
                        all_labels.append(y[criteria_counter].cpu().numpy())
                        all_weights.append(weight)
                        criteria_counter += 1

        all_preds = torch.tensor(np.array(all_preds).flatten(), dtype=torch.float32)
        all_labels = torch.tensor(np.array(all_labels, dtype=int).flatten(), dtype=torch.float32)
        all_weights = torch.tensor(np.array(all_weights), dtype=torch.float32)
        

        if hyper_config['finetune']:
            params = torch.nn.utils.parameters_to_vector(embed_func.model.parameters()).detach()
        else:
            params = torch.tensor([]).to(embed_func.device)
        for model in domain_model_dict.values():
            params = torch.cat([params, torch.nn.utils.parameters_to_vector(model.parameters()).detach()])

        loss, nll, bb_log_prob, clf_log_prob, unweighted_nll = loss_func(all_preds, all_labels, all_weights, params)

        total_loss += loss.item()
        total_nll += nll.item()
        total_bb_log_prob += bb_log_prob.item()
        total_clf_log_prob += clf_log_prob.item()
        total_unweighted_nll += unweighted_nll.item()

    all_preds = all_preds.detach().cpu().numpy()
    all_labels = all_labels.detach().cpu().numpy()
    threshold_preds = (all_preds > 0.5).astype(int).flatten()
    
    metrics = {
        'val_loss': total_loss,
        'val_nll': total_nll,
        'val_bb_log_prob': total_bb_log_prob,
        'val_clf_log_prob': total_clf_log_prob,
        'val_accuracy': accuracy_score(all_labels, threshold_preds),
        'val_precision': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[0],
        'val_recall': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[1],
        'val_auroc': roc_auc_score(all_labels, all_preds),
        'val_auprc': average_precision_score(all_labels, all_preds),
        'val_unweighted_nll': total_unweighted_nll
    }
    return nll, metrics

def train_model(fold, train_dataset, val_dataset,
                embed_func, criteria_embed_func, criteria_combiner,
                domain_model_dict,
                loss_func,
                optimizer,
                hyper_config, problem_config, use_wandb):
    """
    Train a model
    """

    num_domains = problem_config["num_domains"]
    batch_size = hyper_config["batch_size"]
    if batch_size == 'N':
        batch_size = len(train_dataset[0])
    best_val_nll = float('inf')
    
    train_metrics = []
    val_metrics_list = []
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.X, self.y, self.p, self.s = dataset
            self.is_dict = isinstance(self.X, transformers.BatchEncoding)

        def __len__(self):
            if self.is_dict:
                return len(next(iter(self.X.values())))
            else:
                return len(self.X)

        def __getitem__(self, idx):
            if self.is_dict:
                return {k: v[idx] for k, v in self.X.items()}, self.y[idx], self.p[idx], self.s[idx]
            else:
                return self.X[idx], self.y[idx], self.p[idx], self.s[idx]

    train_dataset = (train_dataset[0], torch.tensor(train_dataset[1]).to(embed_func.device), torch.tensor(train_dataset[2]).to(embed_func.device), torch.tensor(train_dataset[3]).to(embed_func.device))
    train_dataset = embed_func.preprocess_data(train_dataset, hyper_config)
    train_dataset = CustomDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = (val_dataset[0], torch.tensor(val_dataset[1]).to(embed_func.device), torch.tensor(val_dataset[2]).to(embed_func.device), torch.tensor(val_dataset[3]).to(embed_func.device))
    val_dataset = embed_func.preprocess_data(val_dataset, hyper_config)
    val_dataset = CustomDataset(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(hyper_config["num_epochs"]):
        epoch_loss = 0
        epoch_nll = 0
        epoch_bb_log_prob = 0
        epoch_clf_log_prob = 0
        epoch_unweighted_nll = 0

        all_preds, all_labels = [], []
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_loss = 0
            X_batch, y_batch, p_batch, s_batch = batch

            all_batch_preds, all_batch_labels, all_batch_weights = [], [], []

            if isinstance(X_batch, dict):
                batch_length = len(next(iter(X_batch.values())))
            else:
                batch_length = len(X_batch)
                
            for i in range(batch_length):
                if isinstance(X_batch, dict):
                    x = {k: v[i].unsqueeze(0) for k, v in X_batch.items()}
                else:
                    x = X_batch[i].unsqueeze(0)
                y = y_batch[i]
                p = p_batch[i]
                s = s_batch[i]

                x_embed = embed_func.forward(x)
                
                criteria_counter = 0
                for d in range(num_domains):
                    num_criteria = problem_config['problems'][p]['domains'][d]["num_criteria"]

                    for c in range(num_criteria):
                        c_embed = criteria_embed_func(torch.tensor([[c]]).to(embed_func.device))

                        final_representation = criteria_combiner(x_embed, c_embed)
                        y_pred = domain_model_dict[d](final_representation)
                        weight = 1.0/num_criteria 

                        all_batch_preds.append(y_pred)
                        all_batch_labels.append(y[criteria_counter])
                        all_batch_weights.append(weight)
                        criteria_counter += 1
            
            #print(all_batch_labels)
            all_batch_preds = torch.stack(all_batch_preds)
            all_batch_labels = torch.stack(all_batch_labels)
            all_batch_weights = torch.tensor(all_batch_weights).to(embed_func.device)
            
            # combine parameters in embedder and all domain models
            if hyper_config['finetune']:
                params = torch.nn.utils.parameters_to_vector(embed_func.model.parameters()).detach()
            else:
                params = torch.tensor([]).to(embed_func.device)
                
            for model in domain_model_dict.values():
                params = torch.cat([params, torch.nn.utils.parameters_to_vector(model.parameters()).detach()])

            batch_loss, batch_nll, batch_bb_log_prob, batch_clf_log_prob, batch_unweighted_nll = loss_func(torch.squeeze(all_batch_preds), all_batch_labels, all_batch_weights, params)
            
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_nll += batch_nll.item()
            epoch_bb_log_prob += batch_bb_log_prob.item()
            epoch_clf_log_prob += batch_clf_log_prob.item()
            epoch_unweighted_nll += batch_unweighted_nll.item()

            all_preds.append(all_batch_preds.detach().cpu().numpy())
            all_labels.append(all_batch_labels.detach().cpu().numpy())
        
        # flatten all_preds and all_labels

        all_preds = np.concatenate(all_preds).flatten()
        # cast labels to int
        all_labels = np.concatenate(all_labels).astype(int).flatten()
        threshold_preds = (all_preds > 0.5).astype(int).flatten()
        train_acc = accuracy_score(all_labels, threshold_preds)
        precision, recall, _, _ = precision_recall_fscore_support(all_labels, threshold_preds, average='macro')
        auroc = roc_auc_score(all_labels, all_preds)  # Keep raw values for AUROC
        auprc = average_precision_score(all_labels, all_preds)  # Keep raw values for AUPRC
        
        train_metrics.append([epoch, epoch_loss, epoch_nll, epoch_bb_log_prob, epoch_clf_log_prob, train_acc, precision, recall, auroc, auprc, epoch_unweighted_nll])
        
        # evaluate every val_freq
        if epoch % hyper_config['val_freq'] == 0:
            with torch.no_grad():
                embed_func.model.eval()
                val_nll, val_metrics = evaluate_model(val_dataloader,hyper_config, embed_func, criteria_embed_func, criteria_combiner, domain_model_dict, loss_func, problem_config, num_domains)
            embed_func.model.train()

            if val_nll < best_val_nll:
                best_val_nll = val_nll
            
            val_metrics_list.append([epoch] + list(val_metrics.values()))
        else:
            val_metrics = {}
        if use_wandb:
            wandb.log({
                'train_loss': epoch_loss,
                'train_nll': epoch_nll,
                'train_bb_log_prob': epoch_bb_log_prob,
                'train_clf_log_prob': epoch_clf_log_prob,
                'train_accuracy': train_acc,
                'train_precision': precision,
                'train_recall': recall,
                'train_auroc': auroc,
                'train_auprc': auprc,
                'train_unweighted_nll': epoch_unweighted_nll,
                **val_metrics
            })
        

        
        print(f'Epoch {epoch} loss: {epoch_loss}')
    
    train_df = pd.DataFrame(train_metrics, columns=['epoch', 'loss', 'nll', 'bb_log_prob',
                                                    'clf_log_prob', 'accuracy', 'precision',
                                                      'recall', 'auroc', 'auprc', 'unweighted_nll'])
    train_df.to_csv(hyper_config['train_metrics_path'].format(fold=fold), index=False)
    
    val_df = pd.DataFrame(val_metrics_list, columns=['epoch', 'loss', 'nll',
                                                      'bb_log_prob', 'clf_log_prob', 'accuracy',
                                                        'precision', 'recall', 'auroc', 'auprc', 'unweighted_nll'])
    val_df.to_csv(hyper_config['val_metrics_path'].format(fold=fold), index=False)

    save_dict = {'domain_model_dict': domain_model_dict}
    if hyper_config['finetune']:
        save_dict['embed_func'] = embed_func
    torch.save(save_dict, hyper_config['final_model_path'].format(fold=fold))

    return

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--hyper_config', type=str, help='Path to hyperparameter configuration file')
    parser.add_argument('--problem_config', type=str, help='Path to problem configuration file')
    args = parser.parse_args()
    
    hyper_config = yaml.load(open(args.hyper_config), Loader=yaml.FullLoader)
    problem_config = yaml.load(open(args.problem_config), Loader=yaml.FullLoader)

    run_experiment(hyper_config, problem_config)