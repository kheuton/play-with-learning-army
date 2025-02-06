from data_loader import load_dataset


def run_experiment(hyper_config, problem_config, data_config):
    """
    Run a single experiment
    :param hyper_config: dict, hyperparameters for the experiment
    :param model_config: dict, model configuration
    :return: float, accuracy of the model
    """

    # Load data
    train_dataset, val_dataset, test_dataset = load_dataset(data_config, hyper_config,
                                                            train=True, val=True, test=True)

    # Train model
    model = train_model(train_dataset, val_dataset, hyper_config, problem_config)

    # Evaluate model
    # should report time taken, precision/recall perclassifier
    # and overall precision/recall
    report_model(model, val_dataset, test_dataset, hyper_config)

    return


def train_model(train_dataset, val_dataset, hyper_config, problem_config):
    """
    Train a model
    :param train_dataset: tuple, (X, y) training data
    :param val_dataset: tuple, (X, y) validation data
    :param hyper_config: dict, hyperparameters for the experiment
    :param model_config: dict, model configuration
    :return: model, trained model
    """

    # This turns a student response into an embedding
    # Astha will write the BOW and frozen BERT embed_func
    # Kyle will write the finetune one
    embed_func = initialize_embedding(hyper_config, train_dataset)
    # This function embeds a criteria, it's probably just the identity
    criteria_embed_func = initialize_criteria_embedding(hyper_config)
    # This function combines the two embeddings, it's probably just concatenation
    criteria_combiner = initialize_combiner(hyper_config)

    # this creates a dictionary of 8 logistic regression predictors
    # key: domain, value: model
    domain_model_dict = initialize_domain_models(hyper_config, problem_config)

    loss_func = initialize_loss(hyper_config)
    optimizer = initialize_optimizer(hyper_config, domain_model_dict, embed_func)

    num_domains = problem_config["num_domains"]
    batch_size = hyper_config["batch_size"]
    # chunk dataset into batches
    train_dataset = batch_dataset(train_dataset, batch_size)

    for epoch in range(hyper_config["num_epochs"]):
        for batch in train_dataset:

            optimizer.zero_grad()
            batch_loss = 0
            X_batch, y_batch, p_batch = batch

            for x, y, p in zip(X_batch, y_batch, p_batch):
                x_embed = embed_func(x)

                for d in range(num_domains):
                    num_criteria = problem_config[p][d]["num_criteria"]

                    for c in range(num_criteria):
                        c_embed = criteria_embed_func(c)
                        final_representation = criteria_combiner(x_embed, c_embed)
                        y_pred = domain_model_dict[d](final_representation)

                        # Compute loss
                        instance_loss = loss_func(y_pred, y)
                        scaled_instance_loss = 1/num_criteria * instance_loss

                        batch_loss += scaled_instance_loss

            # Update model
            batch_loss.backward()
            optimizer.step()

    return