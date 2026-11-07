import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

# ======================================================
# PATH SETUP
# ======================================================
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import get_transforms, load_config
from src.models import ResNet18FineTuning
from src.autoencoder import Autoencoder
from src.grad_cam import GradCAM

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Brain MRI Diagnostic AI",
    page_icon="🧠",
    layout="centered"
)

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = config["class_names"]

# ======================================================
# LOAD MODELS
# ======================================================
@st.cache_resource
def load_models():
    clf = ResNet18FineTuning(num_classes=config["num_classes"]).to(device)
    clf.load_state_dict(torch.load(
        str(PROJECT_ROOT / "models" / "resnet18_finetuning_best_balanced.pth"),
        map_location=device
    ))
    clf.eval()

    ae = Autoencoder().to(device)
    ae.load_state_dict(torch.load(
        str(PROJECT_ROOT / "models" / "autoencoder.pth"),
        map_location=device
    ))
    ae.eval()

    return clf, ae


classifier, autoencoder = load_models()

# ======================================================
# GRAD-CAM
# ======================================================
target_layer = classifier.model.layer4[-1]
gradcam = GradCAM(classifier.model, target_layer)

# ======================================================
# HELPERS
# ======================================================
def preprocess(image):
    transform = get_transforms(train=False)
    return transform(image).unsqueeze(0).to(device)


def predict(x):
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].cpu().numpy()


def anomaly_score(x):
    with torch.no_grad():
        recon = autoencoder(x)
        return torch.mean((x - recon) ** 2).item()


def overlay_cam(image, cam):
    image = np.array(image.resize((128, 128)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# ======================================================
# UI
# ======================================================
st.title("🧠 Brain MRI Diagnostic System")
st.caption("AI-powered tumor classification & anomaly detection")

uploaded = st.file_uploader("Upload MRI scan", type=["png", "jpg", "jpeg"])

# ======================================================
# MAIN PIPELINE
# ======================================================
if uploaded:

    image = Image.open(uploaded).convert("RGB")
    x = preprocess(image)

    # =========================
    # MODEL OUTPUTS
    # =========================
    pred, probs = predict(x)
    confidence = float(np.max(probs))

    error = anomaly_score(x)
    cam = gradcam.generate(x)

    # ======================================================
    # SIMPLE ANOMALY LOGIC
    # ======================================================

    anomaly_state = "NO" if classes[pred].lower() == "notumor" else "YES"

    # ======================================================

    if anomaly_state == "YES":
        final_result = f"⚠️ Abnormal: {classes[pred]}"
    else:
        final_result = f"✅ {classes[pred]}"

    # ======================================================
    # LAYOUT
    # ======================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("MRI Scan")
        st.image(image, use_container_width=True)

        st.subheader("Grad-CAM")
        st.image(overlay_cam(image, cam), use_container_width=True)

    with col2:
        st.subheader("Diagnosis Report")

        st.success(final_result)

        st.markdown("### Anomaly Assessment")
        st.write(f"**Anomaly:** {anomaly_state}")

        st.markdown("### Model Prediction")
        st.metric("Class", classes[pred])
        st.metric("Confidence", f"{confidence:.2%}")

        st.markdown("### Top Predictions")
        top2 = np.argsort(probs)[::-1][:2]
        for i in top2:
            st.write(f"{classes[i]} → {probs[i]:.2%}")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("System Info")
st.sidebar.info(
    """
    • Model: ResNet18 Fine-Tuned  
    • Anomaly: Based on Classifier Prediction  
    • Explainability: Grad-CAM  
    """
)
# ======================================================
# SECOND PAGE: BREAST CANCER MODEL
# ======================================================

from torchvision import models, transforms

# -------------------------
# PAGE SELECTOR
# -------------------------
page = st.sidebar.radio("Select Page", ["Brain MRI", "Breast Cancer"])

# -------------------------
# BREAST MODEL LOADER
# -------------------------
@st.cache_resource
def load_breast_model():
    print("🧬 Loading Breast Cancer model...")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    model_path = PROJECT_ROOT / "models" / "breast_cancer_model.pth"

    print("📁 Loading from:", model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("✅ Breast model loaded")
    return model

breast_model = load_breast_model()

# -------------------------
# BREAST TRANSFORM
# -------------------------
breast_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------------
# BREAST GRAD-CAM (SAFE INIT)
# -------------------------
breast_gradcam = GradCAM(breast_model, breast_model.layer4[-1])

# ======================================================
# BREAST PAGE UI
# ======================================================
if page == "Breast Cancer":

    st.title("🧬 Breast Cancer Detection AI")
    st.caption("Histopathology classification + Grad-CAM")

    uploaded = st.file_uploader(
        "Upload histopathology image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:

        print("📤 Image uploaded")

        image = Image.open(uploaded).convert("RGB")
        x = breast_transform(image).unsqueeze(0).to(device)

        # -------------------------
        # PREDICTION
        # -------------------------
        with torch.no_grad():
            out = breast_model(x)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs).item()

        label = "Cancer" if pred == 1 else "No Cancer"

        st.success(f"🧬 Result: {label} ({conf:.2%})")

        print("🎯 Prediction:", label, conf)

        # -------------------------
        # GRAD-CAM
        # -------------------------
        cam = breast_gradcam.generate(x, pred)

        img_np = np.array(image.resize((128, 128)))

        heatmap = cv2.resize(cam, (128, 128))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # -------------------------
        # DISPLAY
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        with col2:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)

        # -------------------------
        # DETAILS
        # -------------------------
        st.markdown("### Prediction Details")
        st.write(f"Class: **{label}**")
        st.write(f"Confidence: **{conf:.2%}**")