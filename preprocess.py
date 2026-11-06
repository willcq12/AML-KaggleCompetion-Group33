import numpy as np
import pandas as pd
import re
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def split_validation_set(file, random_state=12):
    df = pd.read_csv(file)
    # Hold out the same proportion of each label for validation.
    df_train, df_validation = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df["label"]
    )

    print(f"Train set size: {len(df_train)}")
    print(f"Validation set size: {len(df_validation)}")
    return df_train, df_validation

def unlabel_data(df):
    X = df["text"]
    y = df['label']
    return X, y

def clean_text(text, lowercase=True):
    # Lowercase by default; allow NN experiments to preserve capitalization.
    if lowercase:
        text = text.lower()

    #remove urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # Remove Unix-style paths
    text = re.sub(r'(/[^\s]+)+', ' FILEPATH ', text)

    # Remove Windows-style paths
    text = re.sub(r'[A-Za-z]:\\[^\s]+', ' FILEPATH ', text)

    # Remove long numbers
    text = re.sub(r'\b\d{5,}\b', '', text)

    # Replace UUIDs
    text = re.sub(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
        r'[0-9a-fA-F]{12}\b',
        'UUID',
        text
    )

    # Remove hashes
    text = re.sub(r'\b[a-fA-F0-9]{8,}\b', '', text)

    #normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tfidf_data(X_train, X_validation, X_test):
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


def preprocess_data(featurization=tfidf_data, n=8000, random_state=12):
    
    df_train, df_validation = split_validation_set(_data_file("train.csv"), random_state=random_state)
    
    if n != len(df_train):
        df_train = df_train.sample(n=n, random_state=random_state)

    X_train, y_train = unlabel_data(df_train)
    X_validation, y_validation = unlabel_data(df_validation)

    X_test = pd.read_csv(_data_file("test.csv"))["text"]
    X_train, X_validation, X_test = featurization(X_train, X_validation, X_test)
    return X_train, y_train, X_validation, y_validation, X_test

def nn_data(
    X_train,
    X_validation,
    X_test,
    max_vocab=30000,
    max_length=512,
    preserve_case=False,
):
    """Convert text into padded token IDs, optionally preserving capitalization."""
    if max_vocab < 2 or max_length < 1:
        raise ValueError("max_vocab must be >= 2 and max_length >= 1")

    # Tokenize cleaned text into words and punctuation.
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def tokenize(texts):
        return [
            pattern.findall(clean_text(text, lowercase=not preserve_case))
            for text in texts.fillna("")
        ]

    train_tokens = tokenize(X_train)

    # Learn the vocabulary from training texts only.
    counts = Counter(
        token
        for passage in train_tokens
        for token in passage
    )

    # Reserve 0 for padding and 1 for unknown tokens.
    vocabulary = {
        token: token_id
        for token_id, (token, _) in enumerate(
            counts.most_common(max_vocab - 2),
            start=2,
        )
    }

    def encode(passages):
        sequences = np.zeros(
            (len(passages), max_length),
            dtype=np.int64,
        )

        for row, tokens in enumerate(passages):
            token_ids = [
                vocabulary.get(token, 1)
                for token in tokens[:max_length]
            ]
            sequences[row, :len(token_ids)] = token_ids

        return sequences

    return (
        encode(train_tokens),
        encode(tokenize(X_validation)),
        encode(tokenize(X_test)),
    )

def _data_file(name):
    """Use the same repository data files regardless of working directory."""
    return Path(__file__).resolve().parent / "data" / name


def export_split(file, output_dir, random_state=12):
    """Write the raw shared split without changing the source CSV."""
    output_dir = Path(output_dir)
    train_file = output_dir / "train_split.csv"
    validation_file = output_dir / "validation_split.csv"
    if Path(file).resolve() in {train_file.resolve(), validation_file.resolve()}:
        raise ValueError("The source CSV must not be one of the output files")
    df_train, df_validation = split_validation_set(file, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(train_file, index=False)
    df_validation.to_csv(validation_file, index=False)
    return train_file, validation_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export the shared stratified split")
    parser.add_argument("--export-split", action="store_true", help="write the raw train and validation CSVs")
    parser.add_argument("--input", default=str(Path(__file__).resolve().parent / "data" / "train.csv"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "data" / "splits"))
    parser.add_argument("--random-state", type=int, default=12)
    args = parser.parse_args()
    if args.export_split:
        for path in export_split(args.input, args.output_dir, args.random_state):
            print(f"Saved: {path}")
    else:
        parser.print_help()
