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

loss_func = nn.CrossEntropyLoss()

EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertClassifier().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

best_val_loss = float("inf")
for epoch in range(EPOCHS):
    total_train_loss = 0

    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = loss_func(output, batch["label"].to(device))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epochs: {epoch+1}/{EPOCHS} and avg. training loss = {avg_train_loss:.2f}")

    model.eval()
    total_val_loss = 0
    correct_preds = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            label = batch["label"].to(device)
            output = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_func(output, label)
            total_val_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct_preds += (preds == label).sum().item()
            total += label.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct_preds / total
        print(
            f"Epochs: {epoch+1}/{EPOCHS} and validation loss = {avg_val_loss} | validation accuracy = {val_acc}"
        )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
