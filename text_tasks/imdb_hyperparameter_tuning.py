import optuna
from train_imdb_reviews import IMDBSettings, main as train_imdb_model
from pytorch_lightning.loggers import MLFlowLogger
import os


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    max_seq_len = trial.suggest_int("max_seq_len", 128, 512, step=64)
    transformer_model = trial.suggest_categorical(
        "transformer_model", ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
    )
    max_epochs = trial.suggest_int("max_epochs", 1, 3)
    patience = trial.suggest_int("patience", 2, 5)

    # Update settings
    settings = IMDBSettings(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        transformer_model=transformer_model,
        max_seq_len=max_seq_len,
        patience=patience,
    )

    # Train the model
    val_loss = train_imdb_model(settings)

    return val_loss


if __name__ == "__main__":
    # Ensure MLFlow logging is set up
    experiment_name = "imdb-reviews-hyperparameter-tuning"

    if not os.environ.get("MLFLOW_TRACKING_URI") and os.environ.get(
        "MLFLOW_TRACKING_URL"
    ):
        raise ValueError(
            "Please set the MLFLOW_TRACKING_URI environment variable, hint: URI instead of URL"
        )

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    )

    # Create an Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # Log the best parameters and results to MLFlow
    mlflow_logger.experiment.log_params(study.best_params)
    mlflow_logger.experiment.log_metric("best_val_loss", study.best_value)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
