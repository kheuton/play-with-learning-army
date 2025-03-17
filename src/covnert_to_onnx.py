import torch
import onnx
import yaml
import argparse
from transformers import AutoTokenizer
from bert_model import MultiDomainMultiCriteriaClassifier, tokenize_inputs
from data_loader import load_datasets
from onnxruntime.quantization import quantize_dynamic, QuantType

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert trained model to ONNX')
parser.add_argument('--hyper_config', type=str, required=True, help='Path to hyperparameter configuration file')
parser.add_argument('--problem_config', type=str, required=True, help='Path to problem configuration file')
args = parser.parse_args()

# Load hyperparameters and problem config
with open(args.hyper_config) as f:
    hyper_config = yaml.safe_load(f)

with open(args.problem_config) as f:
    problem_config = yaml.safe_load(f)

# Load datasets (only training dataset is needed for ONNX conversion)
train_datasets, _ = load_datasets(hyper_config, problem_config, train=True, val=True, test=False)

# Select a small batch of real data for ONNX conversion
sample_texts, sample_targets, sample_problem_indices, _, sample_criteria_texts = train_datasets[0][:8]  # Take first 8 samples

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(hyper_config["bert_model_name"])

# Tokenize text inputs
text_inputs = tokenize_inputs(tokenizer, sample_texts)

# Tokenize criteria inputs
criteria_inputs_per_problem = [tokenize_inputs(tokenizer, c_list) for c_list in sample_criteria_texts]

# Convert problem indices to a tensor
problem_indices = torch.tensor(sample_problem_indices, dtype=torch.long)

# Load trained model from saved checkpoint
model_path = hyper_config['final_model_path'].format(fold=0)
model = MultiDomainMultiCriteriaClassifier(
    bert_model_name=hyper_config["bert_model_name"],
    criteria_to_head_mapping=problem_config["criteria_to_head_mapping"],
    output_length=max(len(c_list) for c_list in problem_config["criteria_to_head_mapping"])
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define a wrapper model to make ONNX tracing compatible
class ONNXWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, criteria_input_ids, criteria_attention_mask, problem_indices):
        # Convert back to list of dictionaries
        criteria_inputs_per_problem = [
            {"input_ids": criteria_input_ids[i], "attention_mask": criteria_attention_mask[i]}
            for i in range(len(criteria_input_ids))
        ]
        return self.model({"input_ids": input_ids, "attention_mask": attention_mask}, criteria_inputs_per_problem, problem_indices)

# Wrap model for ONNX tracing
onnx_model = ONNXWrapperModel(model)

# Convert model to ONNX
onnx_path = "bert_model.onnx"
torch.onnx.export(
    onnx_model,
    (text_inputs["input_ids"], text_inputs["attention_mask"], 
     [c["input_ids"] for c in criteria_inputs_per_problem],
     [c["attention_mask"] for c in criteria_inputs_per_problem], problem_indices),
    onnx_path,
    input_names=["input_ids", "attention_mask", "criteria_input_ids", "criteria_attention_mask", "problem_indices"],
    output_names=["predictions", "mask"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "criteria_input_ids": {0: "batch_size"},
        "criteria_attention_mask": {0: "batch_size"},
        "problem_indices": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Model successfully converted to ONNX: {onnx_path}")

# Quantization
quantized_model_path = "bert_model_quantized.onnx"
quantize_dynamic(onnx_path, quantized_model_path, weight_type=QuantType.QInt8)
print(f"Quantized model saved at: {quantized_model_path}")
