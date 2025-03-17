import yaml
import wandb
import torch
import numpy as np
from data_loader import load_datasets, create_hf_dataset
from preprocessing_registry import frozen_preprocessing
from bert_model import MultiDomainMultiCriteriaClassifier
from loss_opt import initialize_loss, initialize_optimizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_experiment(hyper_config, problem_config):
    set_seed(hyper_config['seed'])

    # Initialize wandb if specified
    use_wandb = hyper_config.get('use_wandb', True)
    if use_wandb:
        wandb.init(project=hyper_config['wandb_project'], config=hyper_config, name=hyper_config['experiment_name'])

    # Load datasets
    train_datasets, val_datasets = load_datasets(hyper_config, problem_config, train=True, val=True, test=False)

    criteria_texts = train_datasets[0][-1]
    assert train_datasets[0][-1] == val_datasets[-1][-1], 'Criteria texts must be the same for train and val datasets'
    criteria_to_head_mapping = problem_config['criteria_to_head_mapping']
    
    # Convert each fold to a hf dataset
    train_datasets = [create_hf_dataset(ds)[0] for ds in train_datasets]
    val_datasets = [create_hf_dataset(ds)[0] for ds in val_datasets]

    if hyper_config['embedder'] == 'frozen_bert':
        preprocessor = frozen_preprocessing
    else:
        print(f'OOOPS! Embedder {hyper_config["embedder"]} not supported')

    for fold, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        
        model = MultiDomainMultiCriteriaClassifier(
            bert_model_name=hyper_config['bert_model_name'],
            finetune=hyper_config.get('finetune', False),
            criteria_to_head_mapping=criteria_to_head_mapping,
            output_length=max([len(c_list) for c_list in criteria_to_head_mapping])
        ).to(hyper_config['device'])

        # Initialize loss and optimizer
        criterion = initialize_loss(hyper_config, model)
        optimizer = initialize_optimizer(hyper_config, model)
        import pdb;pdb.set_trace()
        # Preprocess datasets
        train_dataset, train_criteria = preprocessor(train_dataset, criteria_texts, hyper_config)
        val_dataset, val_criteria = preprocessor(val_dataset, criteria_texts, hyper_config)



    

    return



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
