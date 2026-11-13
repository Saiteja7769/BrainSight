from pathlib import Path
import sys
import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from preprocessing import ImageValidationError, load_and_validate_image, preprocess_uploaded_image, make_display_preview
from model import CLASS_NAMES, load_model, predict
from explainability import generate_gradcam, prepare_gradcam_images
from evaluation import TEST_METRICS, CONFUSION_MATRIX

MODEL_PATH = PROJECT_ROOT / "models" / "best_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="BrainSight | MRI Analysis", page_icon="🧠", layout="wide")
st.title("🧠 BrainSight")
st.subheader("Explainable Brain MRI Tumor Classification")
st.write("Upload a brain MRI image to obtain an experimental model prediction and Grad-CAM visualization.")
st.warning(
    "Disclaimer: BrainSight is an experimental AI research project, not a medical diagnostic tool. "
    "Do not use its outputs for diagnosis or clinical decisions. Consult a qualified medical professional."
)

@st.cache_resource(show_spinner="Loading the trained ResNet18 model...")
def get_model(checkpoint_path: str, device_name: str):
    return load_model(checkpoint_path, torch.device(device_name))

if not MODEL_PATH.is_file():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

try:
    model = get_model(str(MODEL_PATH), str(DEVICE))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

st.success("Trained ResNet18 model loaded successfully.")
st.caption(f"Running inference on: {DEVICE}")
st.divider()
st.header("Upload Brain MRI")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["png", "jpg", "jpeg"],
    help="Upload one grayscale brain MRI image."
)

if uploaded_file is not None:
    try:
        image = load_and_validate_image(uploaded_file)
        st.image(make_display_preview(image), caption="Uploaded MRI Scan", width=350)

        input_tensor = preprocess_uploaded_image(image, DEVICE)
        predicted_index, probabilities = predict(model, input_tensor)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index].item() * 100)

        st.divider()
        st.header("Classification Results")
        st.success(f"Predicted Class: {predicted_class}")
        st.metric("Model Confidence (softmax score)", f"{confidence:.2f}%")

        st.subheader("Class-wise Confidence Scores")
        for index, class_name in enumerate(CLASS_NAMES):
            score = float(probabilities[index].item())
            st.write(f"**{class_name}**")
            st.progress(score)
            st.caption(f"{score * 100:.2f}%")
        st.caption("Softmax scores are not calibrated medical probabilities or disease likelihoods.")

        st.divider()
        st.header("Grad-CAM Visualization")
        st.write(
            "Grad-CAM highlights image regions that influenced the selected class score. "
            "It does not identify tumor boundaries or prove tumor location."
        )
        with st.spinner("Generating Grad-CAM explanation..."):
            heatmap = generate_gradcam(model, input_tensor, predicted_index)
            original_rgb, heatmap_color, overlay = prepare_gradcam_images(image, heatmap)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original_rgb, caption="Original MRI", use_container_width=True)
        with col2:
            st.image(heatmap_color, caption="Grad-CAM Heatmap", use_container_width=True)
        with col3:
            st.image(overlay, caption="Explanation Overlay", use_container_width=True)

    except ImageValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")
        st.caption("Try another valid grayscale brain MRI image.")

with st.expander("Model performance on the held-out test set"):
    st.write("Recorded results for the selected baseline ResNet18 checkpoint on the patient-level held-out test set.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{TEST_METRICS['accuracy'] * 100:.2f}%")
    c2.metric("Macro F1", f"{TEST_METRICS['macro_f1'] * 100:.2f}%")
    c3.metric("Macro precision", f"{TEST_METRICS['macro_precision'] * 100:.2f}%")
    c4.metric("Macro recall", f"{TEST_METRICS['macro_recall'] * 100:.2f}%")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(CONFUSION_MATRIX)
    ax.set_xticks(range(3), labels=CLASS_NAMES, rotation=20)
    ax.set_yticks(range(3), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Held-out Test Set")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(CONFUSION_MATRIX[row, col]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "These results are specific to this dataset and split. External validation "
        "across hospitals, scanners, protocols, and populations has not been established."
    )
