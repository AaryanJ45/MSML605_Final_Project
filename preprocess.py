import os
import re
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io

# Boto3 for AWS S3 interaction
import boto3

# Creating a function for text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    return text

# Function to Load data
def load_data(local, bucket=None, file_name=None):
    # If we are running from local computer, load the data from local file system
    if local:
        print("Loading from local")
        df = pd.read_csv(file_name)
    # If we are running from cloud, load the data from S3 bucket
    else:
        print("Loading from S3 bucket")
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=file_name)
        df = pd.read_csv(obj['Body'])
    return df

# Function to save the preprocessed data, call at the end of preprocessing
def save_preprocessed_data(df, local, bucket=None, file_name=None):
    if local:
        print("Saving preprocessed data to local")
        df.to_csv(file_name, index=False)
    else:
        print("Saving preprocessed data to S3 bucket")
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket, Key=file_name, Body=df.to_csv(index=False))

def preprocess_data(args):
    # Loading the data
    df = load_data(local=args.local, bucket=args.bucket, file_name=args.file_name)

    # Doing some basic cleaning
    df = df.dropna(subset=["page_text", "bias"])

    # Normalizing labels
    df["bias"] = df["bias"].replace({
        "leaning-left": "left",
        "leaning-right": "right"
    })

    # Cleaning the page text
    df["page_text"] = df["page_text"].apply(clean_text)

    # Removing the examples that are shorter
    # df = df[df["page_text"].str.len() > 10] Removing for now

    # Encoding the labels
    le = LabelEncoder()
    df["bias"] = le.fit_transform(df["bias"])

    print("Label encoding: ", dict(zip(le.classes_, le.transform(le.classes_))))

    # Train test split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=2, stratify=df["bias"])

    # validation split
    val_df, test_df = train_test_split(train_df, test_size=0.2, random_state=2, stratify=train_df["bias"])

    print("Train set: ", train_df.shape)
    print("Validation set: ", val_df.shape)
    print("Test set: ", test_df.shape)

    # Saving all the outputs
    if args.local:
        save_preprocessed_data(train_df, local=True, file_name="train.csv")
        save_preprocessed_data(val_df, local=True, file_name="val.csv")
        save_preprocessed_data(test_df, local=True, file_name="test.csv")

        os.makedirs("preprocessed_data", exist_ok=True)
        # Saving thte label encoder
        joblib.dump(le, "preprocessed_data/label_encoder.pkl")
    else:
        save_preprocessed_data(train_df, local=False, bucket=args.bucket, file_name="train.csv")
        save_preprocessed_data(val_df, local=False, bucket=args.bucket, file_name="val.csv")
        save_preprocessed_data(test_df, local=False, bucket=args.bucket, file_name="test.csv")

        s3 = boto3.client('s3')
        buffer = io.BytesIO()
        joblib.dump(le, buffer)
        buffer.seek(0)
        s3.put_object(Bucket=args.bucket, Key="preprocessed_data/label_encoder.pkl", Body=buffer.read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=True)
    parser.add_argument("--bucket", type=str, default=None)
    parser.add_argument("--file_name", type=str, default="bias_clean.csv")
    args = parser.parse_args()
    preprocess_data(args)