import numpy as np
import pandas as pd

def split_validation_set(file):
    df = pd.read_csv(file)
    # Randomly sample 80% of the rows for training (8,000 rows)
    df_train = df.sample(frac=0.8, random_state=12)

    # Drop the training rows to get the remaining 20% for testing
    df_validation = df.drop(df_train.index)

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_validation)}")
    return df_train, df_validation

def unlabel_data(df):
    X = df["text"]
    y = df['label']
    return X, y

def preprocess_data(X, y):
    # Preprocessing steps for all data

    return X, y


def sample_labeled_data(file, number_of_samples=8000, random_state=12):
    df_train, df_validation = split_validation_set(file)
    
    df_train = df_train.sample(n=number_of_samples, random_state=random_state)

    X_train, y_train = unlabel_data(df_train)
    X_validation, y_validation = unlabel_data(df_validation)

    X_train, y_train = preprocess_data(X_train, y_train)
    X_validation, y_validation = preprocess_data(X_validation, y_validation)
    return X_train, y_train, X_validation, y_validation
