from data_loader import load_dataset, to_device
from embedder_registry import initialize_embedding, initialize_criteria_embedding, initialize_combiner
from domain_models import initialize_domain_models
from loss_opt import initialize_loss, initialize_optimizer
import torch
import yaml
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_experiment(hyper_config, problem_config):
    """
    Run a single experiment
    """

    set_seed(hyper_config['seed'])
    wandb.init(project=hyper_config['wandb_project'], config=hyper_config, name=hyper_config['experiment_name'])
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_dataset(hyper_config,
                                                            train=True, val=True, test=True)
    
    embedder = initialize_embedding(hyper_config, train_dataset)
    criteria_embedder = initialize_criteria_embedding(hyper_config)
    criteria_combiner = initialize_combiner(hyper_config)

    domain_model_dict = initialize_domain_models(embedder.embedding_size,  embedder.device, problem_config)
    loss_func = initialize_loss(hyper_config, embedder)
    optimizer = initialize_optimizer(hyper_config, domain_model_dict, embedder)

    model = train_model(train_dataset, val_dataset,
                        embedder, criteria_embedder, criteria_combiner,
                        domain_model_dict,
                        loss_func,
                        optimizer,
                        hyper_config, problem_config)
    
    #report_model(model, val_dataset, test_dataset, hyper_config)
    return

def evaluate_model(dataloader, hyper_config, embed_func, criteria_embed_func, criteria_combiner, domain_model_dict, loss_func, problem_config, num_domains):
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            X_batch, y_batch, p_batch, s_batch = batch
            for x, y, p, s in zip(X_batch, y_batch, p_batch, s_batch):
                x, y, p, s = embed_func.preprocess_data((x, y, p, s), hyper_config)
                x_embed = embed_func.forward(x)
                
                criteria_counter = 0
                for d in range(num_domains):
                    num_criteria = problem_config['problems'][p]['domains'][d]["num_criteria"]
                    for c in range(num_criteria):
                        c_embed = criteria_embed_func(torch.tensor([[c]]).to(embed_func.device))
                        final_representation = criteria_combiner(x_embed, c_embed)
                        y_pred = domain_model_dict[d](final_representation)
                        
                        total_loss += loss_func(torch.squeeze(y_pred), y[criteria_counter]).item()
                        all_preds.append(y_pred.detach().cpu().numpy())
                        all_labels.append(y[criteria_counter].cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels, dtype=int).flatten()
    threshold_preds = (all_preds > 0.5).astype(int).flatten()
    
    metrics = {
        'val_loss': total_loss,
        'val_accuracy': accuracy_score(all_labels, threshold_preds),
        'val_precision': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[0],
        'val_recall': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[1],
        'val_auroc': roc_auc_score(all_labels, all_preds),
        'val_auprc': average_precision_score(all_labels, all_preds)
    }
    return total_loss, metrics

def train_model(train_dataset, val_dataset,
                embed_func, criteria_embed_func, criteria_combiner,
                domain_model_dict,
                loss_func,
                optimizer,
                hyper_config, problem_config):
    """
    Train a model
    """

    num_domains = problem_config["num_domains"]
    batch_size = hyper_config["batch_size"]
    best_val_loss = float('inf')
    
    train_metrics = []
    val_metrics_list = []
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.X, self.y, self.p, self.s = dataset

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.p[idx], self.s[idx]

    train_dataset = CustomDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(hyper_config["num_epochs"]):
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch_loss = 0
            X_batch, y_batch, p_batch, s_batch = batch

            for x, y, p, s in zip(X_batch, y_batch, p_batch, s_batch):
                x, y, p, s = embed_func.preprocess_data((x, y, p, s), hyper_config)
                x_embed = embed_func.forward(x)
                
                criteria_counter = 0
                for d in range(num_domains):
                    num_criteria = problem_config['problems'][p]['domains'][d]["num_criteria"]

                    for c in range(num_criteria):
                        c_embed = criteria_embed_func(torch.tensor([[c]]).to(embed_func.device))
                        final_representation = criteria_combiner(x_embed, c_embed)
                        y_pred = domain_model_dict[d](final_representation)

                        instance_loss = loss_func(torch.squeeze(y_pred), y[criteria_counter])
                        scaled_instance_loss = 1/num_criteria * instance_loss
                        batch_loss += scaled_instance_loss
                        
                        all_preds.append(y_pred.detach().cpu().numpy())
                        all_labels.append(y[criteria_counter].cpu().numpy())
                        criteria_counter += 1
                        
            
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels, dtype=int).flatten()
        threshold_preds = (all_preds > 0.5).astype(int).flatten()
        train_acc = accuracy_score(all_labels, threshold_preds)
        precision, recall, _, _ = precision_recall_fscore_support(all_labels, threshold_preds, average='macro')
        auroc = roc_auc_score(all_labels, all_preds)  # Keep raw values for AUROC
        auprc = average_precision_score(all_labels, all_preds)  # Keep raw values for AUPRC
        
        train_metrics.append([epoch, epoch_loss, train_acc, precision, recall, auroc, auprc])
        
        wandb.log({
            'train_loss': epoch_loss,
            'train_accuracy': train_acc,
            'train_precision': precision,
            'train_recall': recall,
            'train_auroc': auroc,
            'train_auprc': auprc
        })
        
        val_loss, val_metrics = evaluate_model(val_dataloader,hyper_config, embed_func, criteria_embed_func, criteria_combiner, domain_model_dict, loss_func, problem_config, num_domains)
        val_metrics_list.append([epoch, val_loss] + list(val_metrics.values()))
        wandb.log(val_metrics)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {'domain_model_dict': domain_model_dict}
            if hyper_config['finetune']:
                save_dict['embed_func'] = embed_func
            torch.save(save_dict, hyper_config['best_model_path'])
        
        print(f'Epoch {epoch} loss: {epoch_loss}')
    
    train_df = pd.DataFrame(train_metrics, columns=['epoch', 'loss', 'accuracy', 'precision', 'recall', 'auroc', 'auprc'])
    train_df.to_csv(hyper_config['train_metrics_path'], index=False)
    
    val_df = pd.DataFrame(val_metrics_list, columns=['epoch', 'loss', 'accuracy', 'precision', 'recall', 'auroc', 'auprc'])
    val_df.to_csv(hyper_config['val_metrics_path'], index=False)

    return

if __name__ == '__main__':
    print('ooops')
    import argparse
    parser = argparse.ArgumentParser(description='Run an experiment')
    parser.add_argument('--hyper_config', type=str, help='Path to hyperparameter configuration file')
    parser.add_argument('--problem_config', type=str, help='Path to problem configuration file')
    args = parser.parse_args()
    
    hyper_config = yaml.load(open(args.hyper_config), Loader=yaml.FullLoader)
    problem_config = yaml.load(open(args.problem_config), Loader=yaml.FullLoader)

    run_experiment(hyper_config, problem_config)