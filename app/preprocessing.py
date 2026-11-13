from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIXTEEN_BIT_MODES = {"I;16", "I;16B", "I;16L", "I"}

class ImageValidationError(ValueError):
    pass

def load_and_validate_image(uploaded_file) -> Image.Image:
    try:
        with Image.open(uploaded_file) as source:
            source.verify()
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            image = image.copy()
    except Exception as exc:
        raise ImageValidationError("The uploaded file is not a readable image.") from exc

    if image.width < 64 or image.height < 64:
        raise ImageValidationError("Image dimensions must be at least 64 × 64 pixels.")
    if image.width * image.height > 25_000_000:
        raise ImageValidationError("Image is too large. Maximum supported size is 25 million pixels.")

    if image.mode not in SIXTEEN_BIT_MODES:
        image = image.convert("L")

    pixels = np.asarray(image, dtype=np.float32)
    if pixels.size == 0 or not np.isfinite(pixels).all():
        raise ImageValidationError("Image contains invalid pixel values.")
    if float(pixels.max()) - float(pixels.min()) <= 1e-8:
        raise ImageValidationError("The image appears blank or has no intensity variation.")
    return image

def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32)
    lower, upper = np.percentile(pixels, [1, 99])
    pixels = np.clip(pixels, lower, upper)
    return ((pixels - lower) / (upper - lower + 1e-8)).astype(np.float32)

def preprocess_uploaded_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = normalize_pixels(np.asarray(image, dtype=np.float32))
    tensor = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)

def make_display_preview(image: Image.Image) -> np.ndarray:
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    pixels = normalize_pixels(np.asarray(resized, dtype=np.float32))
    return np.clip(pixels * 255, 0, 255).astype(np.uint8)
