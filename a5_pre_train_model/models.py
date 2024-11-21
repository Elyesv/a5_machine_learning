import torch
import torch.nn as nn
from torchvision.models import resnet18, efficientnet_b0

def get_model(model_name, num_classes, freeze_layers=True):
    if model_name == "resnet18":
        model = resnet18(pretrained=True)
        if freeze_layers:
            for name, param in model.named_parameters():
                if name.startswith("conv1") or name.startswith("layer1"):
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
                
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),  
            nn.ReLU(),
            nn.Dropout(0.3),              

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),              

            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(pretrained=True)
        if freeze_layers:
            for name, param in model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )
        
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model
