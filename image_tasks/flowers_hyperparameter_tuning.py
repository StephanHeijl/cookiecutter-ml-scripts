import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from dataclasses import dataclass
import optuna

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Device configuration (handled by PyTorch Lightning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing parameters
IMG_SIZE = 224
NUM_CLASSES = 102

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # mean for ImageNet
                         [0.229, 0.224, 0.225])   # std for ImageNet
])

# Load the Flowers-102 dataset
train_set = Flowers102(root='.', split='train', download=True, transform=transform)
val_set = Flowers102(root='.', split='val', download=True, transform=transform)
test_set = Flowers102(root='.', split='test', download=True, transform=transform)

print(
    f"Train set: {len(train_set)} samples\n",
    f"Validation set: {len(val_set)} samples\n",
    f"Test set: {len(test_set)} samples"
)


@dataclass
class FlowerClassifierSettings:
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 25
    patience: int = 5
    model_name: str = 'efficientnet_b1.ra4_e3600_r240_in1k'


# Define the Lightning Module
class FlowerClassifier(pl.LightningModule):
    def __init__(self, settings: FlowerClassifierSettings):
        super(FlowerClassifier, self).__init__()
        # Load the model
        self.model = timm.create_model(
            settings.model_name, 
            pretrained=True
        )
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
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    
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
        print(f"Trainable params: {trainable_params} | All params: {all_params} | "
              f"Trainable%: {100 * trainable_params / all_params:.2f}%")


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    max_epochs = trial.suggest_int('max_epochs', 10, 30)
    patience = trial.suggest_int('patience', 3, 10)
    model_name = trial.suggest_categorical('model_name', [
        'efficientnet_b1.ra4_e3600_r240_in1k',
        'efficientnet_b2.ra4_e3600_r240_in1k',
        'efficientnet_b3.ra4_e3600_r240_in1k'
    ])
    
    # Update settings
    settings = FlowerClassifierSettings(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        model_name=model_name
    )
    
    # Define data loaders with the batch_size from trial
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=4)
    
    # Instantiate the model
    model = FlowerClassifier(settings)
    model.print_trainable_parameters()
    
    # Add callback for early stopping
    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=settings.patience, mode='min')
    
    # Define the Trainer
    trainer = Trainer(
        max_epochs=settings.max_epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[early_stop],
        logger=False,  # Disable logging during hyperparameter search
        log_every_n_steps=1,
    )
    
    model.train()
    
    # Train the model
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        # Handle exceptions during training
        print(f"Exception during training: {e}")
        return None
    
    # Get the validation loss
    val_loss = trainer.callback_metrics.get('val_loss')
    val_acc = trainer.callback_metrics.get('val_acc')
    
    # If val_loss or val_acc is a tensor, convert it to a float
    if isinstance(val_loss, torch.Tensor):
        val_loss = val_loss.item()
    else:
        val_loss = float(val_loss)
    
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.item()
    else:
        val_acc = float(val_acc)
    
    # Report intermediate values to Optuna
    trial.report(val_loss, step=trainer.current_epoch)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    # Return the validation loss
    return val_loss  # Or -val_acc to maximize accuracy


def main(settings=None):
    # Instantiate the model
    if settings is None:
        settings = FlowerClassifierSettings()
    model = FlowerClassifier(settings)
    model.print_trainable_parameters()

    # Define data loaders with the batch_size from settings
    train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=settings.batch_size, shuffle=False, num_workers=4)
    
    # Add callback for early stopping
    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=settings.patience, mode='min')

    # Define the Trainer
    trainer = Trainer(
        max_epochs=settings.max_epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[early_stop],
        logger=False,  # Set to False or configure your logger
        log_every_n_steps=1,
    )

    model.train()

    # Train the model
    trainer.fit(model, train_loader, val_loader)

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

    # Classification report
    print('Classification Report')
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Optionally, log metrics to a logger


if __name__ == '__main__':
    # Create the study
    study = optuna.create_study(direction='minimize')  # minimize val_loss

    # Run the study
    study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the model with the best hyperparameters
    best_settings = FlowerClassifierSettings(
        batch_size=trial.params['batch_size'],
        learning_rate=trial.params['learning_rate'],
        max_epochs=trial.params['max_epochs'],
        patience=trial.params['patience'],
        model_name=trial.params['model_name']
    )
    main(best_settings)
