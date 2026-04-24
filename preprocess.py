import os
import re
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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