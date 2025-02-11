import itertools
import yaml
import os
from pathlib import Path

# Fixed parameters
FIXED_PARAMS = {
    'embedder': 'bow',
    'criteria_embedder': 'identity',
    'combiner': 'concatenate',
    'device': 'cpu',
    'finetune': False,
    'wandb_project': 'sensemaking_bert_bow',
    'seed': 360,
    'num_epochs': 1000,
    'loss': 'bce',
    'use_wandb': False,
    'num_folds': 5
}
# Base output directory
BASE_OUTPUT_DIR = '/cluster/tufts/hugheslab/kheuto01/sensemaking/bow'


# Variable parameters
LEARNING_RATES = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000005, 0.000001]
WEIGHT_DECAYS = [0.1, 0.01, 0.001, 0.0001, 1e-5, 0]
BATCH_SIZES = [32, 65, 128, 'N']
DATA_DIRS = [
    '/cluster/home/kheuto01/code/play-with-learning-army/data/clean/test_15/fold_{fold}/',
    '/cluster/home/kheuto01/code/play-with-learning-army/data/clean/test_20/fold_{fold}'
]


def create_experiment_name(lr, wd, batch_size, data_dir):
    """Create experiment name from parameters including the test number."""
    test_num = '15' if 'test_15' in data_dir else '20'
    return f'test{test_num}_lr{lr}_wd{wd}_bs{batch_size}'

def create_config(lr, weight_decay, batch_size, data_dir):
    """Create a single config dictionary."""
    # Create experiment name
    experiment_name = create_experiment_name(lr, weight_decay, batch_size, data_dir)
    
    # Create output directory path
    data_test_num = 'test_15' if 'test_15' in data_dir else 'test_20'
    output_dir = os.path.join(BASE_OUTPUT_DIR, data_test_num, experiment_name)
    
    # Basic config with fixed parameters
    config = FIXED_PARAMS.copy()
    
    # Add variable parameters
    config.update({
        'learning_rate': lr,
        'opt_weight_decay': weight_decay,
        'batch_size': batch_size,
        'experiment_name': experiment_name,
        
        # Data files
        'train_x_file': os.path.join(data_dir, 'train_x.csv'),
        'train_y_file': os.path.join(data_dir, 'train_y.csv'),
        'val_x_file': os.path.join(data_dir, 'val_x.csv'),
        'val_y_file': os.path.join(data_dir, 'val_y.csv'),
        'test_x_file': os.path.join(data_dir, 'test_x.csv'),
        'test_y_file': os.path.join(data_dir, 'test_y.csv'),
        
        # Metrics and model paths
        'train_metrics_path': os.path.join(output_dir, 'train_metrics_{fold}.csv'),
        'val_metrics_path': os.path.join(output_dir, 'val_metrics_{fold}.csv'),
        'final_model_path': os.path.join(output_dir, 'final_model_{fold}.pth')
    })
    
    return config, output_dir

def main():
    # Generate all combinations of parameters
    combinations = itertools.product(
        LEARNING_RATES,
        WEIGHT_DECAYS,
        BATCH_SIZES,
        DATA_DIRS
    )
    
    # Create configs for each combination
    for lr, weight_decay, batch_size, data_dir in combinations:
        config, output_dir = create_config(lr, weight_decay, batch_size, data_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Write config to file
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created config: {config_path}")

if __name__ == "__main__":
    main()