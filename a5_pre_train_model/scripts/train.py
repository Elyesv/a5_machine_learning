import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import get_dataloaders
from models import get_model
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import matplotlib.pyplot as plt
# import numpy as np

class ImageClassificationModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr=0.001, freeze_layers=False):
        super(ImageClassificationModel, self).__init__()
        self.model = get_model(model_name, num_classes, freeze_layers=freeze_layers)
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True),
            'monitor': 'val_loss',  
        }
        return [optimizer], [scheduler]

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=128, num_workers=4)

    logger = TensorBoardLogger("tb_logs", name="cifar100_resnet18")

    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath="./models",
        filename="finetune_model"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=7,
        verbose=True
    )

    model = ImageClassificationModel(model_name="resnet18", num_classes=100, freeze_layers=True)

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping]
    )

    for name, param in model.model.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")


    trainer.fit(model, train_loader, val_loader)

    torch.save(model.model.state_dict(), "./models/best_model.pth")
    print("Modèle sauvegardé dans ./models/best_model.pth")