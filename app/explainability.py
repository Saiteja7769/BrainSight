import numpy as np
import torch
from PIL import Image

def generate_gradcam(model, input_tensor, class_index: int) -> np.ndarray:
    activations, gradients = {}, {}

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = model.layer4[-1].register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if not 0 <= class_index < logits.shape[1]:
                raise ValueError("Grad-CAM class index is out of range.")
            logits[:, class_index].sum().backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM could not capture model activations.")
        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["value"]).sum(dim=1))[0]
        cam = cam - cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam = cam / maximum
        return cam.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

def prepare_gradcam_images(image: Image.Image, heatmap: np.ndarray):
    import matplotlib.pyplot as plt
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    original = np.asarray(resized, dtype=np.float32)
    low, high = np.percentile(original, [1, 99])
    original = np.clip(original, low, high)
    original = (original - low) / (high - low + 1e-8)
    original_rgb = np.stack([original] * 3, axis=-1)

    heatmap_image = Image.fromarray(np.clip(heatmap * 255, 0, 255).astype(np.uint8))
    heatmap_image = heatmap_image.resize((224, 224), Image.Resampling.BILINEAR)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    heatmap_color = plt.get_cmap("jet")(heatmap_array)[..., :3]
    overlay = np.clip(0.6 * original_rgb + 0.4 * heatmap_color, 0, 1)
    return original_rgb, heatmap_color, overlay
