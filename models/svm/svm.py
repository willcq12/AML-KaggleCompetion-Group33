from sklearn.svm import SVC

from preprocess import preprocess_data, tfidf_data_char, tfidf_data_combo
from scorer import create_kaggle_submission, evaluate_model

def svm(X_train, y_train, random_state=12):
    print("\nSVM")

    model = SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model


X_train, y_train, X_validation, y_validation, X_test = preprocess_data(featurization= tfidf_data_combo)
model = svm(X_train, y_train, random_state=12)
evaluate_model(model, X_validation, y_validation)
# create_kaggle_submission(model, X_test, "submission_svm.csv")