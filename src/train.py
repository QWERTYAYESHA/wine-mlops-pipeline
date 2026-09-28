import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    wine = load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.2,
        random_state=42,
        stratify=wine.target,
    )

    mlflow.set_experiment("wine-classification")

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
