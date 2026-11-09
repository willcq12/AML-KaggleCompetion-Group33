from sklearn.linear_model import LogisticRegression

from preprocess import preprocess_data
from scorer import evaluate_model

def logistic_regression(X_train, X_dev, y_train, y_dev):
    print("\nL2 Logistic Regression")

    model_l2 = LogisticRegression(penalty="l2",
            solver="liblinear")
    model_l2.fit(X_train, y_train)

    return model_l2

X_train, y_train, X_validation, y_validation, X_test = preprocess_data()
model = logistic_regression(X_train, X_validation, y_train, y_validation)
evaluate_model(model, X_validation, y_validation)