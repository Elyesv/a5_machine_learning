import os
import torch
from model import SimpleCNN

# Exporter le modèle en ONNX
def export_to_onnx(model_path, onnx_model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)  # Exemple d'entrée
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to ONNX format at {onnx_model_path}")

# Chemins
model_path = './models/simple_cnn.pth'
onnx_model_path = './models/simple_cnn.onnx'

# Exporter le modèle
export_to_onnx(model_path, onnx_model_path)
