import argparse
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from clearml import Task
from dataset import TextDataset

BATCH = 16
MODEL_SAVE_DIR = "saved_models"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="HuggingFace model ID or shorthand: bert, distilbert",
)
parser.add_argument("--local", action="store_true")
parser.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to saved model directory. Defaults to saved_models/<save_key>.",
)
args = parser.parse_args()

_ALIASES: dict[str, str] = {
    "bert":       "bert",
    "distilbert": "distilbert",
}
_save_key = _ALIASES.get(args.model, args.model.replace("/", "__"))
model_path = args.model_path or os.path.join(MODEL_SAVE_DIR, _save_key)
if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"No saved model found at '{model_path}'. Run train.py first."
    )

le = joblib.load("preprocessed_data/label_encoder.pkl")

test_df = pd.read_csv("test.csv")
X_test = test_df["page_text"].values
y_test = test_df["bias"].values

task = Task.init(
    project_name="Bias Detection",
    task_name=f"test_{_save_key}",
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(le.classes_),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_dataset = TextDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\nTest Results ({args.model})")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

task.get_logger().report_scalar("test/accuracy", "accuracy", accuracy, 0)
task.get_logger().report_scalar("test/precision", "precision", precision, 0)
task.get_logger().report_scalar("test/recall", "recall", recall, 0)
task.get_logger().report_scalar("test/f1", "f1", f1, 0)
