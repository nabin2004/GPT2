import os
from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch

# Load tokenizer and set pad token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load dataset from local text file (ensure file exists)
data_path = os.path.join('.', 'data', 'compiled.txt')
assert os.path.isfile(data_path), f"Dataset not found at {data_path}"

dataset = load_dataset('text', data_files={'train': data_path})

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the input text and return the labels as well (set labels = input_ids for language modeling)
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )

def add_labels(examples):
    examples['labels'] = examples['input_ids']
    return examples

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Save the tokenized dataset
tokenized_datasets.save_to_disk("./tokenized_data")
print("Tokenized dataset saved to './tokenized_data'.")
