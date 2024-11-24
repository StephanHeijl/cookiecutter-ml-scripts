import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from datasets import load_dataset
from functools import partial


# Settings
@dataclass
class IMDBSettings:
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_epochs: int = 3
    transformer_model: str = "bert-base-uncased"
    max_seq_len: int = 512
    patience: int = 3


# Collate function for Hugging Face transformers
def collate_batch(batch, tokenizer, max_seq_len=512):
    labels, texts = zip(*batch)
    label_pipeline = lambda x: 0 if x == "neg" else 1

    labels = torch.tensor([label_pipeline(label) for label in labels])
    tokenized = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return tokenized, labels


# Lightning Module
class IMDBClassifier(pl.LightningModule):
    def __init__(self, settings: IMDBSettings):
        super(IMDBClassifier, self).__init__()
        self.settings = settings
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.transformer_model, num_labels=2
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.transformer_model)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def train_dataloader(self):
        ds = load_dataset("stanfordnlp/imdb")
        train_data = ds["train"]
        return DataLoader(
            train_data,
            batch_size=self.settings.batch_size,
            collate_fn=partial(collate_batch, tokenizer=self.tokenizer),
            shuffle=True,
        )

    def val_dataloader(self):
        ds = load_dataset("stanfordnlp/imdb")
        test_data = ds["test"]
        return DataLoader(
            test_data,
            batch_size=self.settings.batch_size,
            collate_fn=partial(collate_batch, tokenizer=self.tokenizer),
            shuffle=False,
        )

    def training_step(self, batch, batch_idx):
        tokenized, labels = batch
        outputs = self.model(**tokenized)
        loss = self.criterion(outputs.logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        accuracy = (torch.argmax(outputs.logits, dim=1) == labels).float().mean()
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokenized, labels = batch
        outputs = self.model(**tokenized)
        loss = self.criterion(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.settings.learning_rate)


# Main training function
def main(settings=None):
    if settings is None:
        settings = IMDBSettings()

    model = IMDBClassifier(settings)

    # Early stopping
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=settings.patience, mode="min"
    )

    experiment_name = "imdb-reviews-transformer"

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
    mlflow_logger.log_hyperparams(settings.__dict__)

    trainer = Trainer(
        max_epochs=settings.max_epochs,
        devices=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        callbacks=[early_stop],
        logger=mlflow_logger,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    # Return validation loss
    return early_stop.best_score  # Validation loss


if __name__ == "__main__":
    main()
