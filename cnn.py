from copy import deepcopy
from functools import partial
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import nn_data, preprocess_data


SEED = 12
MAX_VOCAB = 30000
MAX_LENGTH = 512
PRESERVE_CASE = True
BATCH_SIZE = 64
EPOCHS = 15
PATIENCE = 3


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        filters=64,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.convolutions = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, kernel_size=width)
            for width in (3, 4, 5)
        ])

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters * 3, 1)

    def forward(self, tokens):
        # (batch, tokens) -> (batch, embedding_dim, tokens)
        embedded = self.embedding(tokens).transpose(1, 2)
        lengths = (tokens != 0).sum(dim=1)

        pooled = []

        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))

            # Exclude windows containing only padding.
            # Keep one window available for an empty passage.
            positions = torch.arange(
                features.size(2),
                device=tokens.device,
            )
            valid = positions[None, :] < lengths.clamp(min=1)[:, None]
            features = features.masked_fill(
                ~valid[:, None, :],
                float("-inf"),
            )

            pooled.append(features.amax(dim=2))

        combined = torch.cat(pooled, dim=1)

        # Return logits; apply sigmoid only when predicting probabilities.
        return self.output(self.dropout(combined)).squeeze(-1)


def make_loader(X, y=None, shuffle=False):
    tensors = [torch.as_tensor(X, dtype=torch.long)]

    if y is not None:
        tensors.append(
            torch.as_tensor(np.asarray(y), dtype=torch.float32)
        )

    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            loss = criterion(model(tokens), labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(tokens)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()

    probabilities = [
        torch.sigmoid(model(batch[0].to(device))).cpu().numpy()
        for batch in loader
    ]

    return np.concatenate(probabilities)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Preserve capitalization: {PRESERVE_CASE}")

    # Pass our sequence settings through the existing preprocessing API.
    featurization = partial(
        nn_data,
        max_vocab=MAX_VOCAB,
        max_length=MAX_LENGTH,
        preserve_case=PRESERVE_CASE,
    )

    X_train, y_train, X_val, y_val, X_test = preprocess_data(
        featurization=featurization,
        random_state=SEED,
    )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test)

    model = TextCNN(vocab_size=MAX_VOCAB).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )

    best_loss = float("inf")
    best_weights = None
    stale_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss = run_epoch(
            model, val_loader, criterion, device
        )

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError("Training produced a non-finite loss.")

        print(
            f"Epoch {epoch:02d} | train loss: {train_loss:.4f} "
            f"| validation loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

            if stale_epochs >= PATIENCE:
                print("Early stopping.")
                break

    # Evaluate and predict using the best epoch.
    model.load_state_dict(best_weights)

    val_probabilities = predict_probabilities(
        model, val_loader, device
    )
    val_predictions = (val_probabilities >= 0.5).astype(int)

    print(f"Accuracy: {accuracy_score(y_val, val_predictions):.4f}")
    print(
        f"F1:       "
        f"{f1_score(y_val, val_predictions, zero_division=0):.4f}"
    )
    print(f"AUROC:    {roc_auc_score(y_val, val_probabilities):.4f}")
    print(f"Log loss: {log_loss(y_val, val_probabilities):.4f}")

    # shuffle=False preserves the original test-row order.
    probabilities = predict_probabilities(
        model, test_loader, device
    )

    data_dir = Path(__file__).resolve().parent / "data"
    test = pd.read_csv(data_dir / "test.csv")

    if len(test) != 3000 or len(probabilities) != len(test):
        raise ValueError(
            "Expected one prediction for each of the 3,000 test rows."
        )

    if not test["id"].is_unique:
        raise ValueError("Test IDs must be unique.")

    valid_probabilities = (
        np.isfinite(probabilities)
        & (probabilities >= 0)
        & (probabilities <= 1)
    )
    if not np.all(valid_probabilities):
        raise ValueError("Predictions must be probabilities between 0 and 1.")

    submission = pd.DataFrame({
        "id": test["id"],
        "label": probabilities,
    })

    output_name = "submission_cnn_case.csv" if PRESERVE_CASE else "submission_cnn.csv"
    output_path = data_dir / output_name
    submission.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
