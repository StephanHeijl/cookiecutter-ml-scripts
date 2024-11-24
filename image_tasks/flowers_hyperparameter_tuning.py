from train_flowers_102_efficientnetv2 import FlowerClassifier, FlowerClassifierSettings
from train_flowers_102_efficientnetv2 import main as train_flowers_102
import optuna


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    max_epochs = trial.suggest_int("max_epochs", 10, 30)
    #patience = trial.suggest_int("patience", 3, 10)
    patience = 5
    model_name = trial.suggest_categorical(
        "model_name",
        [
            "efficientnet_b1.ra4_e3600_r240_in1k",
            "vit_large_patch16_224",
            "rexnet_100"
        ],
    )

    # Update settings
    settings = FlowerClassifierSettings(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        model_name=model_name,
    )

    # Train the model
    val_loss = train_flowers_102(settings)

    return val_loss

if __name__ == "__main__":
    # Create a study
    study = optuna.create_study(direction="minimize")

    # Optimize the hyperparameters
    study.optimize(objective, n_trials=10)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

