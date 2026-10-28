"""Grad-CAM for the locally trained MobileNetV3-Small image classifier.

A localization explanation of model activation, NOT segmentation, causal proof,
or a verified map of disease lesions.
"""
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageOps
import torch
from modules.image_detection import DEFAULT_MODEL, load_detector, image_transform


def explain_leaf(image: Image.Image, model_path=DEFAULT_MODEL, target_label: Optional[str] = None):
    """Return Grad-CAM overlay and heatmap images, class label and model score.

    Model's last convolutional feature block is targeted. Re-loads on demand;
    result is cached by Streamlit session instead of caching model hooks.
    """
    img = ImageOps.exif_transpose(image).convert('RGB')
    if min(img.size) < 64:
        raise ValueError('Please upload an image at least 64×64 pixels.')
    model, classes = load_detector(Path(model_path), device='cpu')
    target_layer = model.features[-1]
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_full_backward_hook(backward_hook)
    try:
        with torch.enable_grad():
            x = image_transform()(img).unsqueeze(0)
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0]
            idx = int(probabilities.argmax().item())
            if target_label is not None:
                if target_label not in classes:
                    raise ValueError('Requested class is not in the trained model.')
                idx = classes.index(target_label)
            model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        if not activations or not gradients:
            raise RuntimeError('Grad-CAM hooks did not receive model activations.')
        acts = activations[-1][0].float()
        grads = gradients[-1][0].float()
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam_np = cam.numpy()
        lo, hi = float(cam_np.min()), float(cam_np.max())
        if hi - lo < 1e-10:
            raise RuntimeError('The network did not produce a spatially informative Grad-CAM map for this image.')
        cam_np = (cam_np - lo) / (hi - lo)
        # Model uses Resize((224,224)), so overlay is displayed on the same resized image.
        base = img.resize((224, 224), Image.Resampling.BILINEAR)
        heat = Image.fromarray(np.uint8(np.clip(cam_np * 255, 0, 255)), mode='L')
        heat = heat.resize(base.size, Image.Resampling.BILINEAR)
        h = np.asarray(heat, dtype=np.float32) / 255.0
        # Visual encoding: blue for low, red/yellow for high activation.
        r = np.clip(1.5 * h, 0, 1)
        g = np.clip(1.5 - np.abs(h - 0.75) * 3, 0, 1)
        b = np.clip(1.0 - 1.4 * h, 0, 1)
        colored = np.uint8(np.stack([r, g, b], axis=-1) * 255)
        original = np.asarray(base).astype(np.float32)
        alpha = (h * 0.47)[..., None]
        overlay = Image.fromarray(np.uint8(np.clip((1-alpha) * original + alpha * colored, 0, 255)))
        return {
            'overlay': overlay, 'heatmap': Image.fromarray(colored),
            'label': classes[idx], 'score_percent': round(float(probabilities[idx].item()*100), 2),
        }
    finally:
        hook_f.remove()
        hook_b.remove()
