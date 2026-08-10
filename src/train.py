from datasets import load_dataset

raw_train = load_dataset("stanfordnlp/imdb", split="train")
test_dataset = load_dataset("stanfordnlp/imdb", split="test")

split_train = raw_train.train_test_split(test_size=0.1, seed=42)
train_dataset = split_train["train"]
val_dataset = split_train["test"]
