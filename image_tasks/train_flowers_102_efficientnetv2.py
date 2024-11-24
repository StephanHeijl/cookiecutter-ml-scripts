import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
import timm
import tempfile
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    matthews_corrcoef,
)
from dataclasses import dataclass

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

# Device configuration (handled by PyTorch Lightning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing parameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 102

# Preprocessing transforms
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],  # mean for ImageNet
            [0.229, 0.224, 0.225],
        ),  # std for ImageNet
    ]
)

# Load the Flowers-102 dataset
train_set = Flowers102(root=".", split="train", download=True, transform=transform)
val_set = Flowers102(root=".", split="val", download=True, transform=transform)
test_set = Flowers102(root=".", split="test", download=True, transform=transform)

print(
    f"Train set: {len(train_set)} samples\n",
    f"Validation set: {len(val_set)} samples\n",
    f"Test set: {len(test_set)} samples",
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


@dataclass
class FlowerClassifierSettings:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 5
    patience: int = 5
    model_name: str = "efficientnet_b1.ra4_e3600_r240_in1k"
    run_on_test_set: bool = False


# Define the Lightning Module
class FlowerClassifier(pl.LightningModule):
    def __init__(self, settings: FlowerClassifierSettings):
        super(FlowerClassifier, self).__init__()
        # Load the EfficientNetV2 model
        self.model = timm.create_model(settings.model_name, pretrained=True)
        self.settings = settings

        # Adjust the classifier to match the number of classes (102)
        num_features = self.model.get_classifier().in_features

        self.model.classifier = torch.nn.Linear(num_features, NUM_CLASSES)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.parameters(), lr=self.settings.learning_rate)
        optimizer.train()
        return optimizer

    def print_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params} | All params: {all_params} | "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )


def run_model_on_test_set(model):
    # Evaluate on the test set
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return y_true, y_pred


def main(settings=None):
    # Instantiate the model
    if settings is None:
        settings = FlowerClassifierSettings()
    model = FlowerClassifier(settings)
    model.print_trainable_parameters()

    # Add callback for early stopping
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=settings.patience, mode="min"
    )

    experiment_name = "flowers-102-efficientnetv2"

    if not os.environ.get("MLFLOW_TRACKING_URI") and os.environ.get(
        "MLFLOW_TRACKING_URL"
    ):
        raise ValueError(
            "Please set the MLFLOW_TRACKING_URI environment variable, hint: URI instead of URL"
        )

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        tags={"dev": "sheijl"},
    )
    mlflow_logger.log_hyperparams(settings.__dict__)

    print(mlflow_logger.run_id, mlflow_logger._tracking_uri)

    # Define the Trainer
    trainer = Trainer(
        max_epochs=settings.max_epochs,
        devices=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        callbacks=[early_stop],
        logger=mlflow_logger,
        log_every_n_steps=1,
    )

    model.train()

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    if settings.run_on_test_set:
        y_true, y_pred = run_model_on_test_set(model)

        # Classification report
        print("Classification Report")
        print(classification_report(y_true, y_pred, digits=4))

        # Confusion matrix
        print("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        # Log classification report to MLFlow
        mlflow_logger._mlflow_client.log_text(
            text=classification_report(y_true, y_pred, digits=4),
            artifact_file="classification_report.txt",
            run_id=mlflow_logger.run_id,
        )

        # Log test metrics to MLFlow
        mlflow_logger.log_metrics(
            {
                "test_accuracy": accuracy_score(y_true, y_pred),
                "test_matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
                "test_loss": model.criterion(outputs, labels).item(),
            }
        )

        return {
            "test_accuracy": accuracy_score(y_true, y_pred),
            "test_matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
            "val_loss": early_stop.best_score,
        }
    else:
        return early_stop.best_score  # Validation loss


if __name__ == "__main__":
    main()
