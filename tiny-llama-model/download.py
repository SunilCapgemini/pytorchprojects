from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fine-tune a causal language model")
parser.add_argument("--model-name", type=str, required=True, help="Name of the pre-trained model")
args = parser.parse_args()

model_name = args.model_name

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the ./data folder
model.save_pretrained(f"./data/{model_name}")
tokenizer.save_pretrained(f"./data/{model_name}")