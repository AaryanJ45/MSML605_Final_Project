import os
import re
import logging
import joblib
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional

logger = logging.getLogger(__name__)

MAX_LEN = 256
MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}

# In-memory cache: key = "bert" or "distilbert"
_cache: dict[str, dict] = {}


def _clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _download_from_s3(bucket: str, model_key: str, local_dir: str) -> None:
    """Sync a S3 prefix into a local directory."""
    s3 = boto3.client("s3")
    prefix = f"saved_models/{model_key}/"
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            relative = key[len(prefix):]
            if not relative:
                continue
            dest = os.path.join(local_dir, relative)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            logger.info("Downloading s3://%s/%s → %s", bucket, key, dest)
            s3.download_file(bucket, key, dest)


def _download_label_encoder_from_s3(bucket: str, local_path: str) -> None:
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    logger.info("Downloading label encoder from s3://%s/preprocessed_data/label_encoder.pkl", bucket)
    s3.download_file(bucket, "preprocessed_data/label_encoder.pkl", local_path)


def load_model(
    model_key: str,
    model_save_dir: str,
    label_encoder_path: str,
    bucket: Optional[str] = None,
) -> None:
    """Load model into in-memory cache. Downloads from S3 if not present locally."""
    if model_key in _cache:
        return

    local_model_dir = os.path.join(model_save_dir, model_key)

    if not os.path.isdir(local_model_dir) or not os.listdir(local_model_dir):
        if not bucket:
            raise RuntimeError(
                f"Model '{model_key}' not found at '{local_model_dir}' and no BUCKET configured."
            )
        logger.info("Model not found locally — downloading from S3 bucket '%s'", bucket)
        _download_from_s3(bucket, model_key, local_model_dir)

    if not os.path.isfile(label_encoder_path):
        if not bucket:
            raise RuntimeError(
                f"Label encoder not found at '{label_encoder_path}' and no BUCKET configured."
            )
        _download_label_encoder_from_s3(bucket, label_encoder_path)

    logger.info("Loading tokenizer and model from '%s'", local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    le = joblib.load(label_encoder_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_dir,
        num_labels=len(le.classes_),
    )
    model.eval()

    _cache[model_key] = {"tokenizer": tokenizer, "model": model, "le": le}
    logger.info("Model '%s' loaded and cached.", model_key)


def predict(text: str, model_key: str) -> dict:
    """Run inference on a single text. Model must be loaded first via load_model()."""
    if model_key not in _cache:
        raise RuntimeError(f"Model '{model_key}' is not loaded. Call load_model() first.")

    cached = _cache[model_key]
    tokenizer = cached["tokenizer"]
    model = cached["model"]
    le = cached["le"]

    cleaned = _clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())
    label = le.inverse_transform([pred_idx])[0]
    confidence = round(float(max(probs)), 4)
    probabilities = {cls: round(float(p), 4) for cls, p in zip(le.classes_, probs)}

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
    }
