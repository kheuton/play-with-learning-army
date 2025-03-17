import torch
import yaml
import wandb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from bert_model import MultiDomainMultiCriteriaClassifier, tokenize_inputs, compute_loss
from transformers import AutoTokenizer
from data_loader import load_datasets
from loss_opt import initialize_loss, initialize_optimizer
from torch.utils.data import DataLoader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, bert_embeddings, labels, problem_indices, criteria_embeddings):
        self.bert_embeddings = bert_embeddings  # List of embeddings
        self.labels = labels  # List of integers
        self.problem_indices = problem_indices  # List of integers
        self.criteria_embeddings = criteria_embeddings  # List of lists of embeddings

    def __getitem__(self, idx):
        return (
            self.bert_embeddings[idx], 
            self.labels[idx], 
            self.problem_indices[idx], 
            self.criteria_embeddings[idx]  # <- Ensure this is a **list of lists** at this stage
        )

    def __len__(self):
        return len(self.labels)

def preprocess_data(dataset, tokenizer):
    """Tokenizes texts and criteria in advance."""

    texts, targets, problem_indices, prediction_counts, criteria_texts = dataset
    
    tokenized_texts = [tokenize_inputs(tokenizer, [text]) for text in texts]
    
    tokenized_criteria = [[tokenize_inputs(tokenizer, single_criteria) for single_criteria in c_list] for c_list in criteria_texts]

    return MyDataset(tokenized_texts, torch.tensor(targets, dtype=torch.float32), torch.tensor(problem_indices, dtype=torch.long), tokenized_criteria)

def custom_collate_fn(batch):
    """Collate function for tokenized inputs."""
    #print(batch)

    
    text_inputs, targets, problem_indices, criteria_inputs = zip(*batch)
    # Convert lists of dictionaries into batch tensors
    #batch_text_inputs = {k: torch.cat([d[k] for d in text_inputs], dim=0) for k in text_inputs[0]}
    #import pdb; pdb.set_trace()
    #batch_criteria_inputs = [{k: torch.cat([d[k] for d in crit_input], dim=0) for k in crit_input[0].keys()} for crit_input in criteria_inputs]
    
    return text_inputs, torch.stack(targets), torch.stack(problem_indices), criteria_inputs

def move_inputs_to_device(text_inputs, criteria_inputs, problem_indices, device):
    """Moves tokenized inputs to the specified device."""
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    criteria_inputs = [{k: v.to(device) for k, v in crit_input.items()} for crit_input in criteria_inputs]
    problem_indices = problem_indices.to(device)
    return text_inputs, criteria_inputs, problem_indices

def run_experiment(hyper_config, problem_config):
    set_seed(hyper_config['seed'])

    # Initialize wandb if specified
    use_wandb = hyper_config.get('use_wandb', True)
    if use_wandb:
        wandb.init(project=hyper_config['wandb_project'], config=hyper_config, name=hyper_config['experiment_name'])

    # Load datasets
    train_datasets, val_datasets = load_datasets(hyper_config, problem_config, train=True, val=True, test=False)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hyper_config['bert_model_name'])
    criteria_to_head_mapping = problem_config['criteria_to_head_mapping']

    train_datasets = [preprocess_data(ds, tokenizer) for ds in train_datasets]
    val_datasets = [preprocess_data(ds, tokenizer) for ds in val_datasets]

    model = MultiDomainMultiCriteriaClassifier(
        bert_model_name=hyper_config['bert_model_name'],
        finetune=hyper_config.get('finetune', False),
        criteria_to_head_mapping=criteria_to_head_mapping,
        output_length=max([len(c_list) for c_list in criteria_to_head_mapping])
    ).to(hyper_config['device'])

    # Initialize loss and optimizer
    criterion = initialize_loss(hyper_config, model)
    optimizer = initialize_optimizer(hyper_config, model)

    for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        train_model(
            fold, train_dataset, val_dataset,
            model, tokenizer, criterion, optimizer,
            hyper_config, problem_config, use_wandb
        )

def tokenize_in_chunks(tokenizer, texts, chunk_size):
    return [tokenizer(texts[i:i+chunk_size], padding=True, truncation=True, return_tensors='pt') for i in range(0, len(texts), chunk_size)]


def train_model(fold, train_dataset, val_dataset, model, tokenizer, criterion, optimizer, hyper_config, problem_config, use_wandb):
    batch_size = hyper_config['batch_size']
    num_epochs = hyper_config['num_epochs']
    if batch_size == 'N':
        batch_size = len(train_dataset)
    # Convert tuple of lists into a list of tuples for proper shuffling
    #train_dataset = list(zip(*train_dataset)) if isinstance(train_dataset, tuple) else train_dataset
    #val_dataset = list(zip(*val_dataset)) if isinstance(val_dataset, tuple) else val_dataset
    #

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    best_val_loss = float('inf')

    train_metrics_list = []
    val_metrics_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:

            texts, targets, problem_indices, criteria_inputs = batch
            chunk_size = hyper_config.get('chunk_size', 32) if model.finetune else len(texts)
            #text_chunks = tokenize_in_chunks(tokenizer, texts, chunk_size)
            #text_chunks = texts
            text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            problem_chunks = [problem_indices[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            criteria_chunks  = [criteria_inputs[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            #problem_chunks = problem_indices
            #criteria_chunks = criteria_inputs
            #criteria_chunks  = [tokenize_inputs(tokenizer, c_list) for c_list in criteria_inputs]
            #criteria_chunks = [criteria_chunks[i:i+chunk_size]for i in range(0, len(texts), chunk_size)]
            #problem_chunks = [problem_indices[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            
            #import pdb; pdb.set_trace();
            predictions_list, mask_list = [], []
            for text_chunk, criteria_chunk, problem_chunk in zip(text_chunks, criteria_chunks, problem_chunks):

                #text_chunk, criteria_chunk, problem_chunk = move_inputs_to_device(
                #    text_chunk, criteria_chunk, problem_chunk, hyper_config['device']
                #)
                
                predictions, mask = model(
                    text_inputs=text_chunk,
                    criteria_inputs_per_problem=criteria_chunk,
                    problem_indices=problem_chunk,
                )
                predictions_list.append(predictions)
                mask_list.append(mask)
            predictions = torch.cat(predictions_list, dim=0)
            mask = torch.cat(mask_list, dim=0)
            targets = targets.to(hyper_config['device'])
            params = torch.nn.utils.parameters_to_vector(model.parameters())
            loss = compute_loss(predictions, targets, mask, criterion, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch) % hyper_config['val_freq'] == 0:
            val_loss, val_metrics = evaluate_model(val_loader, model, tokenizer, criterion, hyper_config)

            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **val_metrics
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Store metrics
            train_metrics_list.append({'epoch': epoch + 1, 'train_loss': train_loss})
            val_metrics_list.append({'epoch': epoch + 1, 'val_loss': val_loss, **val_metrics})

            

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), hyper_config['final_model_path'].format(fold=fold))

    train_df = pd.DataFrame(train_metrics_list)
    val_df = pd.DataFrame(val_metrics_list)
    
    train_csv_path = hyper_config['train_metrics_path'].format(fold=fold)
    val_csv_path = hyper_config['val_metrics_path'].format(fold=fold)

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)


def evaluate_model(dataloader, model, tokenizer, criterion, hyper_config):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            texts, targets, problem_indices, student_id, criteria_inputs = batch

            # Tokenize and move inputs to the correct device
            texts, targets, problem_indices, student_id, criteria_inputs = batch
            chunk_size = hyper_config.get('chunk_size', 32) if model.finetune else len(texts)
            text_chunks = tokenize_in_chunks(tokenizer, texts, chunk_size)
            criteria_chunks  = [tokenize_inputs(tokenizer, c_list) for c_list in criteria_inputs]
            criteria_chunks = [criteria_chunks[i:i+chunk_size]for i in range(0, len(texts), chunk_size)]
            problem_chunks = [problem_indices[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
            
            predictions_list, mask_list = [], []
            for text_chunk, criteria_chunk, problem_chunk in zip(text_chunks, criteria_chunks, problem_chunks):

                text_chunk, criteria_chunk, problem_chunk = move_inputs_to_device(
                    text_chunk, criteria_chunk, problem_chunk, hyper_config['device']
                )
                
                predictions, mask = model(
                    text_inputs=text_chunk,
                    criteria_inputs_per_problem=criteria_chunk,
                    problem_indices=problem_chunk,
                )
                predictions_list.append(predictions)
                mask_list.append(mask)
            predictions = torch.cat(predictions_list, dim=0)
            mask = torch.cat(mask_list, dim=0)

            targets = targets.to(hyper_config['device'])
            params = torch.nn.utils.parameters_to_vector(model.parameters())
            loss = compute_loss(predictions, targets, mask, criterion, params)
            val_loss += loss.item()

            numpy_mask = mask.cpu().numpy()
            masked_predictions = predictions.cpu().numpy()[numpy_mask == 1]
            masked_targets = targets.cpu().numpy()[numpy_mask == 1]
            all_preds.extend(masked_predictions.flatten())
            all_labels.extend(masked_targets.flatten())

    threshold_preds = (np.array(all_preds) > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(all_labels, threshold_preds),
        'precision': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[0],
        'recall': precision_recall_fscore_support(all_labels, threshold_preds, average='macro')[1],
        'auroc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds)
    }

    return val_loss, metrics

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with MultiDomainMultiCriteriaClassifier')
    parser.add_argument('--hyper_config', type=str, help='Path to hyperparameter configuration file')
    parser.add_argument('--problem_config', type=str, help='Path to problem configuration file')
    args = parser.parse_args()

    with open(args.hyper_config) as f:
        hyper_config = yaml.safe_load(f)

    with open(args.problem_config) as f:
        problem_config = yaml.safe_load(f)

    run_experiment(hyper_config, problem_config)
