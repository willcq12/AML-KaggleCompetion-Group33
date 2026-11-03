from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def evaluate_model(model, X, y):
    # Generate predicted class labels
    y_pred = model.predict(X)

    # Generate prediction probabilities for AUROC
    y_prob = model.predict_proba(X)[:, 1]

    # Calculate metrics
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    auroc = roc_auc_score(y, y_prob)

    # Print results
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")