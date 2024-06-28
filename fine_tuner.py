import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, \
    TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.data import Dataset

# Check if a GPU is available and if not, use a mps
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps")
print(f'Using device: {device}')

import os

# Specify the path to the input text file
input_file = "input_text_cleaned.txt" # text
output_file = "output_text.txt"

def is_hidden(filepath):
    return os.path.basename(filepath).startswith('.')

with open(output_file, "w") as outfile:
    with open(input_file) as infile:
        for line in infile:
            # only write the line if it's not empty (and not just whitespace)
            if line.strip():
                outfile.write(line)


## GPT-2 Small ('gpt2'): 124 million parameters.
## GPT-2 Medium ('gpt2-medium'): 345 million parameters.
## GPT-2 Large ('gpt2-large'): 774 million parameters.
## GPT-2 XL ('gpt2-xl'): 1.5 billion parameters.


# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Your custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        with open(file_path, "r") as f:
            self.text = f.read().splitlines()
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        tokenized_inputs = self.tokenizer(
            self.text[idx],
            truncation=True,
            padding="max_length",
            max_length=128,#you can increase 
            return_tensors="pt")
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs

# Load data
data = CustomDataset(tokenizer, output_file, 128)

# Create a data collator that will dynamically pad the sequences
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments and Trainer
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=2000, # Increse for more training from the fine-tuning data
    learning_rate=1e-6,  # Decrease the learning rate for smaller fine-tuning data
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=200,
    load_best_model_at_end=False,
    evaluation_strategy="no",
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    eval_dataset=None,  # You can specify an evaluation dataset here
    data_collator=data_collator,  # Add the data collator here
)

trainer.train()

