import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import time

class MultiDomainMultiCriteriaClassifier(nn.Module):
    def __init__(self, 
                 criteria_to_head_mapping: list,    # List of lists: Maps each criterion to one of the 8 heads
                 embedding_dim: int = 768,
                 num_heads: int = 8,
                 output_length: int = 16,
                 finetune: bool = True,
                 bert_model_name: str = None):
        super(MultiDomainMultiCriteriaClassifier, self).__init__()

        if bert_model_name is not None:
            # Load BERT model
            self.bert = AutoModel.from_pretrained(bert_model_name)

            self.finetune = finetune
            if not self.finetune:
                # Freeze all BERT layers:
                for param in self.bert.parameters():
                    param.requires_grad = False
        

        # Create 8 shared classification heads with sigmoid activation
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()  # Apply sigmoid after the linear layer for binary predictions
            ) for _ in range(num_heads)
        ])

        self.output_length = output_length

        # Map each criterion to a classification head
        self.criteria_to_head_mapping = criteria_to_head_mapping  # Shape: [problems][criteria_indices]

    def forward(self, dataset, criteria):
        
        batch_size = len(dataset['embedding'])
        outputs = []
        lengths = []


        for i in range(batch_size):
            problem_idx = dataset['problem_indices'][i]
            text_embs = dataset['embedding'][i] 

            criteria_indices = self.criteria_to_head_mapping[problem_idx]

            # Predict for each criterion using its mapped head
            problem_outputs = []

            for j, head_idx in enumerate(criteria_indices):
                combined_emb = text_embs + criteria[int(problem_idx)][j]  # Combine embeddings
                prediction = self.classification_heads[head_idx](combined_emb)  # Sigmoid applied inside head
                problem_outputs.append(prediction.squeeze())

            outputs.append(torch.stack(problem_outputs))
            lengths.append(len(problem_outputs))

        # Pad outputs to uniform length for efficient loss computation
        max_length = self.output_length
        padded_outputs = torch.zeros(batch_size, max_length, device=text_embs.device)
        mask = torch.zeros(batch_size, max_length, device=text_embs.device)

        for i, output in enumerate(outputs):
            length = lengths[i]
            if length > 0:
                padded_outputs[i, :length] = output
                mask[i, :length] = 1  # Mark valid predictions

        return padded_outputs, mask  # (batch_size, max_criteria), (batch_size, max_criteria)


def tokenize_inputs(tokenizer, texts, max_length=512):
    """Tokenizes a list of texts for efficient batching."""
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)


def compute_loss(predictions, targets, mask, criterion, params):
    """
    Computes masked loss, ensuring padded values do not affect training.

    Args:
        predictions (torch.Tensor): Model outputs (batch_size, max_criteria).
        targets (torch.Tensor): Ground truth labels (batch_size, max_criteria).
        mask (torch.Tensor): Binary mask indicating valid predictions.
        criterion (nn.Module): Loss function (e.g., nn.BCELoss(reduction='none')).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    loss, nll, bb_log_prob, clf_log_prob, unweighted_nll = criterion(predictions, targets, params)

    masked_loss = (loss * mask).sum() / mask.sum()  # Average over valid predictions
    final_loss = masked_loss + bb_log_prob + clf_log_prob

    return final_loss


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    bert_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # Example criteria texts per problem
    criteria_texts_per_problem = [
        ["Criterion 1 for Problem 1", "Criterion 2 for Problem 1", "Criterion 3 for Problem 1"],
        ["Criterion 1 for Problem 2", "Criterion 2 for Problem 2", "Criterion 3 for Problem 2", "Criterion 4 for Problem 2"],
        ["Criterion 1 for Problem 3", "Criterion 2 for Problem 3", "Criterion 3 for Problem 3"],
        ["Criterion 1 for Problem 4", "Criterion 2 for Problem 4", "Criterion 3 for Problem 4", "Criterion 4 for Problem 4", "Criterion 5 for Problem 4"]
    ]

    # Map criteria to classification heads
    criteria_to_head_mapping = [
        [0, 1, 2],
        [1, 2, 3, 4],
        [2, 3, 4],
        [3, 4, 5, 6, 7]
    ]

    # Tokenize criteria for each problem
    criteria_inputs_per_problem = [tokenize_inputs(tokenizer, criteria_texts) for criteria_texts in criteria_texts_per_problem]

    # Tokenize input texts
    texts = ["Response for problem 1.", "Response for problem 2.", "Response for problem 4."]
    text_inputs = tokenize_inputs(tokenizer, texts)

    # Problem indices and prediction counts
    problem_indices = [0, 1, 3]  # Corresponding problems

    # Ground truth labels (padded to max_criteria across the batch)
    targets = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 1.0]
    ])

    # Initialize model
    model = MultiDomainMultiCriteriaClassifier(
        bert_model_name=bert_model_name,
        criteria_to_head_mapping=criteria_to_head_mapping,
        output_length=targets.size(1)
    )

    # Forward pass
    predictions, mask = model(
        text_inputs=text_inputs,
        criteria_inputs_per_problem=criteria_inputs_per_problem,
        problem_indices=problem_indices
    )

    # Loss computation
    criterion = nn.BCELoss(reduction='none')
    loss = compute_loss(predictions, targets, mask, criterion)

    print("Predictions:", predictions)
    print("Mask:", mask)
    print("Loss:", loss.item())
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")
