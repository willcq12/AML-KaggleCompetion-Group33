import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer

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

def clean_text(text):
    #lowercase all text
    text = text.lower()

    #stop words??

    #punctioatin???

    #lemantize??

    #numbers?

    #file paths?

    #urls?

    #normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

def preprocess_data(X_train, X_validation, X_test):
    #tf-idf
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_clean = X_train.apply(clean_text)
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)

    X_validation_clean = X_validation.apply(clean_text)
    X_validation_tfidf = vectorizer.transform(X_validation_clean)

    X_test_clean = X_test.apply(clean_text)
    X_test_tfidf = vectorizer.transform(X_test_clean)

    return X_train_tfidf, X_validation_tfidf, X_test_tfidf


def sample_labeled_data(file, number_of_samples=8000, random_state=12):
    df_train, df_validation = split_validation_set(file)
    
    df_train = df_train.sample(n=number_of_samples, random_state=random_state)

    X_train, y_train = unlabel_data(df_train)
    X_validation, y_validation = unlabel_data(df_validation)

    X_test = pd.read_csv("test.csv")["text"]
    X_train, X_validation, X_test = preprocess_data(X_train, X_validation, X_test)
    return X_train, y_train, X_validation, y_validation, X_test
