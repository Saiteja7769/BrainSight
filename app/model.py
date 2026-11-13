from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model

def load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model_state_dict' entry.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model

def predict(model, input_tensor):
    model.eval()
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
    return int(probabilities.argmax().item()), probabilities
