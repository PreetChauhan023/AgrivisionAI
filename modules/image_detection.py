"""ML-based leaf disease inference. No heuristic diagnosis fallback."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'artifacts' / 'leaf_disease_mobilenet.pt'


def _network(num_classes: int):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def image_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_detector(path=DEFAULT_MODEL, device='cpu'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No trained image model at {path}. Run train_image_model.py first.')
    # Load only trusted locally trained artifacts; never unpickle unknown downloads.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    classes = payload['classes']
    model = _network(len(classes))
    model.load_state_dict(payload['state_dict'])
    model.to(device).eval()
    return model, classes


def analyse_crop_image(pil_image: Image.Image, model_path=DEFAULT_MODEL, device='cpu'):
    model, classes = load_detector(model_path, device)
    img = ImageOps.exif_transpose(pil_image).convert('RGB')
    if min(img.size) < 64:
        raise ValueError('Image resolution is too small (minimum 64x64).')
    with torch.inference_mode():
        logits = model(image_transform()(img).unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1]
    return {'disease': classes[int(order[0])], 'confidence': float(probabilities[order[0]]*100),
            'predictions': [{'disease': classes[int(i)], 'probability': float(probabilities[i])}
                            for i in order[:min(5, len(classes))]],
            'needs_review': bool(probabilities[order[0]] < 0.70),
            'model_type': 'MobileNetV3-Small (trained CNN)',
            'note': 'Softmax scores are not calibrated confidence or diagnostic certainty.'}
