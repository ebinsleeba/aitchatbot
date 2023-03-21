import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader

# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define the dataset
def read_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def encode_dataset(dataset, tokenizer, max_length=512):
    return [tokenizer.encode(text)[:max_length-1] for text in dataset]

# Load the dataset and create data loader
train_dataset = encode_dataset(read_dataset("conversation.txt"), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Fine-tune the model on the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(5):
    for batch in train_loader:
        input_ids = torch.tensor(batch).to(device)
        target_ids = input_ids[:, 1:].clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100
        outputs = model(input_ids, labels=target_ids)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("conversation_model")