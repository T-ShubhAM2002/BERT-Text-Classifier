import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_len=512):
        self.text = hf_dataset["text"]
        self.label = hf_dataset["label"]
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = int(self.label[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            max_length=self.max_len,  # Max BERT length
            padding="max_length",  # Pad short reviews to max_length
            truncation=True,  # Truncate long reviews to max_length
            return_tensors="pt",  # Return PyTorch Tensors
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }
