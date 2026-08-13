import torch
from datasets import load_dataset
from dataset import IMDBDataset
from torch.utils.data import DataLoader
from model import BertClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

test_dataset = load_dataset("stanfordnlp/imdb", split="test")
test_data = IMDBDataset(test_dataset, tokenizer)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

all_preds = []
all_labels = []
model = BertClassifier().to(device)

model.load_state_dict(
    torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
)
model.eval()

with torch.no_grad():
    for batch in test_loader:
        label = batch["label"].to(device)
        output = model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

print(
    classification_report(all_labels, all_preds, target_names=["Positive", "Negative"])
)
