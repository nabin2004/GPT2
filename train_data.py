import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Load tokenized dataset
tokenized_datasets = load_from_disk("/kaggle/input/gpttt2/New folder/tokenized_data")

# Fast debugging: use a small dataset; comment out for full training
tokenized_datasets["train"] = tokenized_datasets["train"].select(range(2000))

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    overwrite_output_dir=True,
    num_train_epochs=2,                         # Fewer epochs for speed, increase later if needed
    per_device_train_batch_size=16,             # Higher batch size if GPU allows
    gradient_accumulation_steps=1,              # 1 if you can fit full batch into memory
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=5e-5,                         # Slightly higher LR for faster convergence
    logging_dir='./logs',
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="no",
    logging_first_step=True,
    fp16=torch.cuda.is_available(),             # Use mixed precision if available
    gradient_checkpointing=False,               # Disable if not memory constrained (faster)
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

# Train!
trainer.train()

# Save the final model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Training complete. Model saved to './fine_tuned_model'.")
