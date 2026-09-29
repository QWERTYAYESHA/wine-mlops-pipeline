import mlflow
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_and_split_data


def main():
    # Load the test split
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Load the registered champion model
    model_uri = "models:/WineClassifier@champion"
    model = mlflow.sklearn.load_model(model_uri)

    # Make predictions on the test set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Calculate final test metrics
    test_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    test_log_loss = log_loss(
        y_test,
        probabilities,
    )

    print("WineClassifier @ champion")
    print(f"Test Macro F1: {test_macro_f1:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Log Loss: {test_log_loss:.4f}")


if __name__ == "__main__":
    main()
