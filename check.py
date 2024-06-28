import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your fine-tuned GPT-2 model and tokenizer
model_path = "results/checkpoint-124000" #path checkpoint
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Move the model to the specified device
model.to(device)

# Create a prompt text for the model to complete
prompt_text = "Жазында"

# Tokenize the prompt text and convert to tensor
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
attention_mask = tokenizer(prompt_text, return_tensors="pt").attention_mask

# Move input_ids and attention_mask tensors to the same device as the model
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Generate text from the model
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pad_token_id=tokenizer.pad_token_id,
    max_length=200,
    num_beams=5,
    temperature=1.5,
    top_k=50,
    do_sample=True  # Enable sampling to consider temperature setting
)

# Decode the generated text back to a string
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

