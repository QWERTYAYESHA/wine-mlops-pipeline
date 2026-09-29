import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from src.data import load_and_split_data


def evaluate_model(model, X_train, y_train, model_name, config):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_f1_scores = []
    val_f1_scores = []
    train_accuracy_scores = []
    val_accuracy_scores = []
    train_log_loss_scores = []
    val_log_loss_scores = []

    for train_index, val_index in cv.split(X_train, y_train):
        X_fold_train = X_train[train_index]
        X_fold_val = X_train[val_index]
        y_fold_train = y_train[train_index]
        y_fold_val = y_train[val_index]

        model.fit(X_fold_train, y_fold_train)

        train_predictions = model.predict(X_fold_train)
        val_predictions = model.predict(X_fold_val)

        train_probabilities = model.predict_proba(X_fold_train)
        val_probabilities = model.predict_proba(X_fold_val)

        train_f1_scores.append(
            f1_score(y_fold_train, train_predictions, average="macro")
        )
        val_f1_scores.append(
            f1_score(y_fold_val, val_predictions, average="macro")
        )

        train_accuracy_scores.append(
            accuracy_score(y_fold_train, train_predictions)
        )
        val_accuracy_scores.append(
            accuracy_score(y_fold_val, val_predictions)
        )

        train_log_loss_scores.append(
            log_loss(y_fold_train, train_probabilities)
        )
        val_log_loss_scores.append(
            log_loss(y_fold_val, val_probabilities)
        )

    train_f1 = sum(train_f1_scores) / len(train_f1_scores)
    val_f1 = sum(val_f1_scores) / len(val_f1_scores)

    train_accuracy = sum(train_accuracy_scores) / len(train_accuracy_scores)
    val_accuracy = sum(val_accuracy_scores) / len(val_accuracy_scores)

    train_loss = sum(train_log_loss_scores) / len(train_log_loss_scores)
    val_loss = sum(val_log_loss_scores) / len(val_log_loss_scores)

    # Train final model on the complete training dataset
    model.fit(X_train, y_train)

    # Create MLflow signature and input example
    input_example = X_train[:1]
    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)

    with mlflow.start_run() as run:
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("milestone", "3")

        mlflow.log_param("model_family", model_name)

        for parameter, value in config.items():
            mlflow.log_param(parameter, value)

        mlflow.log_metric("train_macro_f1", train_f1)
        mlflow.log_metric("validation_macro_f1", val_f1)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("validation_accuracy", val_accuracy)

        mlflow.log_metric("train_log_loss", train_loss)
        mlflow.log_metric("validation_log_loss", val_loss)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        run_id = run.info.run_id

    print(f"{model_name} - {config}")
    print(f"Train Macro F1: {train_f1:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Train Log Loss: {train_loss:.4f}")
    print(f"Validation Log Loss: {val_loss:.4f}")
    print(f"Run ID: {run_id}")
    print("-" * 50)

    return run_id, val_f1


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    mlflow.set_experiment("Wine-Cultivar-Classification")

    random_forest_configs = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": None},
    ]

    gradient_boosting_configs = [
        {"n_estimators": 100, "learning_rate": 0.05},
        {"n_estimators": 150, "learning_rate": 0.10},
        {"n_estimators": 200, "learning_rate": 0.15},
    ]

    for config in random_forest_configs:
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
        )

        evaluate_model(
            model,
            X_train,
            y_train,
            "RandomForestClassifier",
            config,
        )

    for config in gradient_boosting_configs:
        model = GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            random_state=42,
        )

        evaluate_model(
            model,
            X_train,
            y_train,
            "GradientBoostingClassifier",
            config,
        )


if __name__ == "__main__":
    main()
