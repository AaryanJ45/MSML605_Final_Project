import argparse
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from clearml import Task
from dataset import TextDataset, MAX_LEN

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["bert", "distilbert"], required=True)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()

if args.model == "bert":
    MODEL_NAME = "bert-base-uncased"
else:
    MODEL_NAME = "distilbert-base-uncased"

#all configurations
THRESHOLD = 0.80
BATCH = 16
EPOCHS = 3
LR = 2e-5
MODEL_SAVE_DIR = "saved_models"

# Load preprocessed splits produced by preprocess.py
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df["page_text"].values
y_train = train_df["bias"].values
X_test = test_df["page_text"].values
y_test = test_df["bias"].values

le = joblib.load("preprocessed_data/label_encoder.pkl")

#initializing clearml
task = Task.init(
    project_name="Bias Detection",
    task_name="bert_vs_distilbert"
)

#tokenizing text
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_)
)
train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range (EPOCHS):
    model.train()

    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

save_path = os.path.join(MODEL_SAVE_DIR, args.model)
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to {save_path}")