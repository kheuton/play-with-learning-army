from transformers import AutoTokenizer, AutoModel
import torch

def frozen_preprocessing(dataset, criteria_texts, hyper_config):
    
    if hyper_config['batch_size'] == 'N':
        batch_size = len(dataset)
    else:
        batch_size = hyper_config['batch_size']

    tokenizer = AutoTokenizer.from_pretrained(hyper_config['bert_model_name'])
    bert = AutoModel.from_pretrained(hyper_config['bert_model_name'])
    bert.eval()

    def tokenize(data_row):
        return tokenizer(data_row, padding='max_length', return_tensors='pt', truncation=True, max_length=512)


    tokenized_dataset = dataset.map(lambda x: tokenize(x['text']),
                                    batched=True, batch_size=batch_size)
    
    tokenized_criteria = {problem_id: [tokenizer(crit_text,
                                                 padding='max_length',
                                                 return_tensors='pt',
                                                 truncation=True,
                                                max_length=512)
                                        for crit_text in criteria_texts[problem_id]]
                            for problem_id in criteria_texts.keys()}
    

    # Define function to apply BERT
    def process_with_bert(example):
        # Extract tokenized inputs
        inputs = {
            "input_ids": torch.tensor(example["input_ids"]), 
            "attention_mask": torch.tensor(example["attention_mask"]),
        }
        
        # Include token_type_ids if available
        if "token_type_ids" in example:
            inputs["token_type_ids"] = torch.tensor(example["token_type_ids"])

        # Run through BERT (disable gradient computation)
        with torch.no_grad():
            outputs = bert(**inputs)

        # Extract last hidden state (CLS token embedding)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # Shape: (hidden_dim)

        # Preserve other dataset features and add the BERT embedding
        example["bert_embedding"] = cls_embedding
        return example

    embedded_dataset = tokenized_dataset.map(process_with_bert, batched=True, batch_size=batch_size)

    embedded_criteria = {problem_id: [bert(**crit_text) for crit_text in tokenized_criteria[problem_id]]
                            for problem_id in tokenized_criteria.keys()}

    return embedded_dataset, embedded_criteria
