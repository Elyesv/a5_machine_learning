import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from pytorch_lightning.utilities import rank_zero_only

class SimpleCNNLightning(pl.LightningModule):
    def __init__(self):
        super(SimpleCNNLightning, self).__init__()
        self.model = SimpleCNN()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

def prepare_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader

test_loader = prepare_test_data()

model_path = os.path.join('./models', 'simple_cnn.pth')
model = SimpleCNNLightning()

if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.model.load_state_dict(torch.load(model_path, map_location='cpu'))
else:
    raise FileNotFoundError(f"No pre-trained model found at {model_path}")

trainer = pl.Trainer(
    accelerator='auto',  
    devices=1,
    logger=False,        
    enable_checkpointing=False,
    enable_model_summary=False,
    callbacks=[],
    max_epochs=1
)

trainer.test(model, dataloaders=test_loader)
