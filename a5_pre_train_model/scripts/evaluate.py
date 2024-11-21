import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import get_dataloaders
from models import get_model
import pytorch_lightning as pl

class ImageClassificationEvaluator(pl.LightningModule):
    def __init__(self, model_name, num_classes, model_path):
        super(ImageClassificationEvaluator, self).__init__()
        self.model = get_model(model_name, num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.criterion = torch.nn.CrossEntropyLoss()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

if __name__ == "__main__":
    _, test_loader = get_dataloaders(batch_size=64, num_workers=4)

    model_path = "./models/modelv4.pth" 

    evaluator = ImageClassificationEvaluator(model_name="resnet18", num_classes=100, model_path=model_path)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False
    )

    trainer.test(evaluator, test_loader)
