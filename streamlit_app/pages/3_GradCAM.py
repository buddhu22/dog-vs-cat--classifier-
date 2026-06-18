import streamlit as st
import tempfile
import sys
import numpy as np
import cv2
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.predict import Predictor
from src.gradcam import GradCAM
from src.utils import load_and_preprocess_image
from src.config import IMAGE_SIZE
from streamlit_app.styles import inject_global_css, render_sidebar_branding

st.set_page_config(page_title="Grad-CAM", page_icon="", layout="wide")

# Inject global CSS and sidebar
inject_global_css()
render_sidebar_branding()

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Grad-CAM Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Peer inside the CNN\'s mind. Grad-CAM highlights the exact regions the model focused on to arrive at its decision — '
    'bringing transparency to deep learning.'
    '</div>',
    unsafe_allow_html=True,
)

# Info banner
st.markdown(
    """
    <div class="info-banner">
        <strong>What is Grad-CAM?</strong><br>
        Gradient-weighted Class Activation Mapping uses the gradients flowing into the final convolutional layer
        to produce a coarse localization map highlighting important regions in the image for the predicted class.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Load model & GradCAM (cached) ──────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return Predictor()

@st.cache_resource
def load_gradcam(_predictor):
    """Cache the GradCAM object so it isn't re-created on every rerun."""
    return GradCAM(_predictor.model)

predictor = load_predictor()

gradcam = None
if predictor.model is not None:
    try:
        gradcam = load_gradcam(predictor)
    except Exception as e:
        st.error(f"Failed to initialize Grad-CAM: {e}")

# ── File uploader & Controls ───────────────────────────────────────────────
col_upload, col_controls = st.columns([2, 1], gap="large")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload an image for explainability analysis...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
    )

with col_controls:
    st.markdown('<div style="padding-top: 1rem;">', unsafe_allow_html=True)
    alpha = st.slider("Overlay Transparency (Alpha)", min_value=0.1, max_value=0.9, value=0.4, step=0.1)
    st.caption("Adjust how much of the original image is visible under the heatmap.")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and gradcam is not None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = Path(tmp_file.name)

        with st.spinner("Generating activation heatmap..."):
            # Preprocess image once
            img_array = load_and_preprocess_image(tmp_path, target_size=IMAGE_SIZE)

            # Predict using already-preprocessed array
            predicted_class, confidence = predictor.predict_image_array(img_array)

            # Generate heatmap
            class_idx = 1 if predicted_class == "Dog" else 0
            heatmap = gradcam.generate_heatmap(img_array, class_index=class_idx)

            # Overlay heatmap with selected alpha
            original_img, overlayed_img = gradcam.overlay_heatmap(tmp_path, heatmap, alpha=alpha)
            
            # Generate raw heatmap image for middle panel
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            raw_heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            raw_heatmap_color = cv2.cvtColor(raw_heatmap_color, cv2.COLOR_BGR2RGB)

        # ── Prediction Result Card ──
        icon = "" if predicted_class == "Dog" else ""
        cls = "dog" if predicted_class == "Dog" else "cat"
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:1rem; margin-top: 0;">
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;">Prediction</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#fbbf24' if cls=='dog' else '#a78bfa'};">{icon} {predicted_class}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;">Confidence</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#fbbf24' if cls=='dog' else '#a78bfa'};">{confidence:.1f}%</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;">Target Class</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #e5e7eb;">Index {class_idx}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #9ca3af; text-transform: uppercase;">Conv Layer Used</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #60a5fa;">{gradcam.last_conv_layer_name}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Three-panel images ──
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align: center; margin-bottom: 0.5rem;"><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);">Original Image</span></div>',
                unsafe_allow_html=True,
            )
            st.image(original_img, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align: center; margin-bottom: 0.5rem;"><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3);">Raw Heatmap</span></div>',
                unsafe_allow_html=True,
            )
            st.image(raw_heatmap_color, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="glass-card" style="padding: 1rem;">', unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align: center; margin-bottom: 0.5rem;"><span style="display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3);">Grad-CAM Overlay</span></div>',
                unsafe_allow_html=True,
            )
            st.image(overlayed_img, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Legend & interpretation ──
        st.markdown(
            """
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 16px; padding: 1.2rem 1.5rem; margin-top: 1rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #e5e7eb; margin-bottom: 10px; letter-spacing: 0.5px;">How to Read the Heatmap</div>
                <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.88rem; color: #d1d5db;">
                    <div style="width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; background: linear-gradient(135deg, #ef4444, #f59e0b);"></div>
                    <span><strong>Red / Yellow (warm)</strong> — High activation. The model focused heavily on these regions to make its decision.</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.88rem; color: #d1d5db;">
                    <div style="width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; background: linear-gradient(135deg, #22c55e, #06b6d4);"></div>
                    <span><strong>Green / Cyan</strong> — Moderate activation. Partially influential regions.</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.88rem; color: #d1d5db;">
                    <div style="width: 14px; height: 14px; border-radius: 4px; flex-shrink: 0; background: linear-gradient(135deg, #3b82f6, #1e3a5f);"></div>
                    <span><strong>Blue / Dark (cool)</strong> — Low activation. These regions had minimal impact on the classification.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()

elif uploaded_file is None:
    # ── Empty state ──
    st.markdown("")
    st.markdown(
        """
        <div class="upload-hint">
            <div class="upload-icon"></div>
            <div class="upload-hint-text">
                Upload an image above to visualize which regions<br>
                the CNN focuses on when making its classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
