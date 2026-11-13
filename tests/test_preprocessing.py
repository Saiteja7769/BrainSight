import io
import numpy as np
import pytest
import torch
from PIL import Image

from app.preprocessing import ImageValidationError, load_and_validate_image, preprocess_uploaded_image

def test_preprocessing_shape_type_and_finite_values():
    pixels = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    image = Image.fromarray(pixels)
    tensor = preprocess_uploaded_image(image, torch.device("cpu"))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()

def test_blank_image_is_rejected():
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(buffer, format="PNG")
    buffer.seek(0)
    with pytest.raises(ImageValidationError, match="blank"):
        load_and_validate_image(buffer)

def test_16_bit_png_is_preserved():
    pixels = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    buffer.seek(0)
    loaded = load_and_validate_image(buffer)
    assert loaded.mode in {"I;16", "I;16B", "I;16L", "I"}
    assert float(np.asarray(loaded, dtype=np.float32).max()) > 255
