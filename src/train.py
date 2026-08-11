import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from dataset import IMDBDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model import BertClassifier

raw_train = load_dataset("stanfordnlp/imdb", split="train")
test_dataset = load_dataset("stanfordnlp/imdb", split="test")

split_train = raw_train.train_test_split(test_size=0.1, seed=42)
train_dataset = split_train["train"]
val_dataset = split_train["test"]

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

train_data = IMDBDataset(train_dataset, tokenizer)
val_data = IMDBDataset(val_dataset, tokenizer)
test_data = IMDBDataset(test_dataset, tokenizer)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

model = BertClassifier()
loss_func = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(EPOCHS):
    loss = 0
    total_loss = 0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = loss_func(output, batch["label"])
        optimizer.step()
        total_loss += loss.item()
    print(f"Epochs: {epoch+1} and training loss = {total_loss/len(train_loader)}")
