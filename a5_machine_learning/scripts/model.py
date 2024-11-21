import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #32x32 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #32x32 -> 32x32
        self.pool1 = nn.MaxPool2d(2, 2) #32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) #16x16 -> 8x8
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) #8x8 -> 4x4
        
        self.bath_norm1 = nn.BatchNorm2d(64)
        self.bath_norm2 = nn.BatchNorm2d(128)
        self.bath_norm3 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.bath_norm1(x)
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.bath_norm2(x)
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.bath_norm3(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
