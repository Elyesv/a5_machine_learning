import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = SimpleCNN().to(device)
model_path = os.path.join('./models', 'simple_cnn.pth')

if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
else:
    print("No pre-trained model found. Starting from scratch.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):  
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

os.makedirs('./models', exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
