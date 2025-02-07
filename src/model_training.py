from data_loader import load_dataset, to_device
from embedder_registry import initialize_embedding, initialize_criteria_embedding, initialize_combiner
from domain_models import initialize_domain_models
from loss_opt import initialize_loss, initialize_optimizer
import torch
import yaml

def run_experiment(hyper_config, problem_config):
    """
    Run a single experiment
    :param hyper_config: dict, hyperparameters for the experiment
    :param model_config: dict, model configuration
    :return: float, accuracy of the model
    """

    # Load data
    train_dataset, val_dataset, test_dataset = load_dataset(hyper_config,
                                                            train=True, val=True, test=True)
    
    # This turns a student response into an embedding
    # Astha will write the BOW and frozen BERT embed_func
    # Kyle will write the finetune one
    embedder = initialize_embedding(hyper_config, train_dataset)
    # This function embeds a criteria, it's probably just the identity
    criteria_embedder = initialize_criteria_embedding(hyper_config)
    # This function combines the two embeddings, it's probably just concatenation
    criteria_combiner = initialize_combiner(hyper_config)

    #train_dataset = embedder.preprocess_data(train_dataset, hyper_config)
    #val_dataset = embedder.preprocess_data(val_dataset, hyper_config)
    #test_dataset = embedder.preprocess_data(test_dataset, hyper_config)

    # this creates a dictionary of 8 logistic regression predictors
    # key: domain, value: model
    domain_model_dict = initialize_domain_models(embedder.embedding_size,  embedder.device, problem_config)

    loss_func = initialize_loss(hyper_config)
    optimizer = initialize_optimizer(hyper_config, domain_model_dict, embedder)

    # Train model
    model = train_model(train_dataset, val_dataset,
                        embedder, criteria_embedder, criteria_combiner,
                        domain_model_dict,
                        loss_func,
                        optimizer,
                        hyper_config, problem_config)

    # Evaluate model
    # should report time taken, precision/recall perclassifier
    # and overall precision/recall
    report_model(model, val_dataset, test_dataset, hyper_config)

    return


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
    # chunk dataset into batches
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.X, self.y, self.p, self.s = dataset

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.p[idx], self.s[idx]

    train_dataset = CustomDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(hyper_config["num_epochs"]):
        epoch_loss = 0
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

                        # Compute loss
                        instance_loss = loss_func(torch.squeeze(y_pred), y[criteria_counter])
                        scaled_instance_loss = 1/num_criteria * instance_loss

                        batch_loss += scaled_instance_loss
                        criteria_counter += 1

            # Update model
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        print(f'Epoch {epoch} loss: {epoch_loss}')
            

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