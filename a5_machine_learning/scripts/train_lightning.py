import os
import torch
import lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

class SimpleCNNLightning(pl.LightningModule):
    def __init__(self):
        super(SimpleCNNLightning, self).__init__()
        self.model = SimpleCNN()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)

def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

train_loader, val_loader = prepare_data()

model_path = os.path.join('./models', 'simple_cnn.pth')
model = SimpleCNNLightning()

if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.model.load_state_dict(torch.load(model_path, map_location='cpu'))

logger = TensorBoardLogger("tb_logs", name="SimpleCNN")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto', 
    devices=1,
    log_every_n_steps=10,
    logger=logger
)

trainer.fit(model, train_loader, val_loader)

os.makedirs('./models', exist_ok=True)
torch.save(model.model.state_dict(), model_path)
print(f"Model saved to {model_path}")
