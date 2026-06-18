import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from streamlit_app.styles import inject_global_css, render_sidebar_branding
from src.config import ASSETS_DIR

st.set_page_config(
    page_title="Bio-Inspired Dog vs Cat Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    inject_global_css()
    render_sidebar_branding()
    
    st.markdown('<div class="hero-title"> Cat vs Dog Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Production-Grade Deep Learning Pipeline for Image Classification with Interpretability.'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # Hero Banner
    banner_path = ASSETS_DIR / "hero_banner.jpg"
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)
        
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ── Feature Highlight Cards ──
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Real-Time Predictions")
        st.markdown(
            "Upload any image and experience **sub-second inference latency**. "
            "The system leverages an optimized Convolutional Neural Network architecture "
            "to extract spatial hierarchies and classify subjects with high confidence."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Grad-CAM Explainability")
        st.markdown(
            "Go beyond black-box predictions. Using **Gradient-weighted Class Activation Mapping**, "
            "the system highlights the exact pixel regions that influenced the model's decision, "
            "ensuring transparency and trust."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Robust Analytics")
        st.markdown(
            "Deep dive into the model's empirical performance across thousands of test samples. "
            "Explore comprehensive metrics including **ROC curves**, **Precision-Recall**, "
            "and interactive **Confusion Matrices**."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ── Dataset & Architecture Summary ──
    st.markdown("### Project Overview")
    col_data, col_arch = st.columns([1, 1], gap="large")
    
    with col_data:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Dataset Highlights")
        st.markdown("""
        - **Total Images:** 25,000 (Kaggle Dogs vs Cats)
        - **Training Split:** 20,000 images
        - **Validation/Test Split:** 5,000 images
        - **Balance:** 50% Dogs | 50% Cats
        - **Augmentation:** Configured for robust generalization
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_arch:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Model Architecture Summary")
        st.markdown("""
        - **Input Layer:** 256x256x3 RGB Tensors
        - **Feature Extraction:** 3x Sequential Conv2D + MaxPooling blocks
        - **Regularization:** Batch Normalization & Dropout (0.1)
        - **Classification Head:** Dense layers converging to a single Sigmoid neuron
        - **Optimizer:** Adam with Binary Cross-Entropy
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info("Navigate using the sidebar to explore Predictions, Grad-CAM, and Analytics.")

if __name__ == "__main__":
    main()
