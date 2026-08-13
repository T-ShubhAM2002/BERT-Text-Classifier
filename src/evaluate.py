import torch
from model import BertClassifier
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
all_preds = []
all_labels = []
model = BertClassifier()
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()
