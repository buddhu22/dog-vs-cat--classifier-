import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from streamlit_app.styles import inject_global_css, render_sidebar_branding
from src.config import ASSETS_DIR

st.set_page_config(page_title="About", page_icon="", layout="wide")

inject_global_css()
render_sidebar_branding()

# ── Header ──
st.markdown('<div class="hero-title hero-title-about">About the Developer & Project</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Developer Card ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    # Adding photo of a cute dog as requested
    dog_path = ASSETS_DIR / "cute_dog.png"
    if dog_path.exists():
        st.image(str(dog_path), width="stretch", caption="Honorary Co-Developer ")
    else:
        st.image("https://images.unsplash.com/photo-1543852786-1cf6624b9987", width="stretch")

with col2:
    st.markdown('<div class="dev-name" style="font-size: 1.8rem; font-weight: 800; color: #e5e7eb;">Abhay Mishra</div>', unsafe_allow_html=True)
    st.markdown('<div class="dev-role" style="font-size: 1rem; color: #60a5fa; font-weight: 600; text-transform: uppercase; margin-bottom: 1rem;">Machine Learning Engineer</div>', unsafe_allow_html=True)

    st.markdown("""
    This application was built as part of a comprehensive machine learning portfolio project,
    showcasing end-to-end ML skills from model training to production deployment.
    """)

    # Skill badges
    skills = [
        ("", "Deep Learning", "#8b5cf6"),
        ("", "Computer Vision", "#3b82f6"),
        ("", "Explainable AI", "#06b6d4"),
        ("", "Model Deployment", "#10b981"),
        ("", "OOP Python", "#f59e0b"),
        ("", "Data Analytics", "#ef4444"),
    ]
    badges_html = ""
    for icon, name, color in skills:
        badges_html += (
            f'<span style="display:inline-block; padding:6px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; margin:4px 4px; border:1px solid {color}33; background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12); color:{color};">'
            f'{icon} {name}</span>'
        )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Social links
    st.markdown(
        """
        <a href="https://www.linkedin.com/in/abhay-mishra-640286365?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" style="display:inline-flex; align-items:center; gap:8px; padding:10px 22px; border-radius:12px; font-size:0.95rem; font-weight:600; text-decoration:none; margin:6px 8px 6px 0; background:linear-gradient(135deg, #0077b5, #005f8d); color:#fff; box-shadow:0 4px 16px rgba(0, 119, 181, 0.3);">
            LinkedIn
        </a>
        <a href="https://github.com/buddhu22" target="_blank" style="display:inline-flex; align-items:center; gap:8px; padding:10px 22px; border-radius:12px; font-size:0.95rem; font-weight:600; text-decoration:none; margin:6px 8px 6px 0; background:linear-gradient(135deg, #333, #1a1a2e); color:#fff; box-shadow:0 4px 16px rgba(0, 0, 0, 0.3);">
            GitHub
        </a>
        """,
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Project Documentation ──
st.markdown("### Project Documentation")

doc_col1, doc_col2 = st.columns(2, gap="large")

with doc_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Problem Statement")
    st.markdown("""
    The objective of this project is to build a robust, scalable image classification pipeline capable of distinguishing between dogs and cats. 
    It aims to demonstrate a complete MLOps lifecycle: from dataset processing and model training to explainable inference and an interactive web deployment.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Dataset Details")
    st.markdown("""
    - **Source:** Kaggle Dogs vs Cats Dataset
    - **Total Images:** 25,000 RGB images
    - **Training Set:** 20,000 images (80%)
    - **Test Set:** 5,000 images (20%)
    - **Class Balance:** Perfectly balanced (50/50 split)
    - **Preprocessing:** Resized to 256x256, normalized to [0,1]
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Training Strategy")
    st.markdown("""
    - **Optimizer:** Adam Optimizer
    - **Loss Function:** Binary Cross-Entropy
    - **Epochs:** 10
    - **Batch Size:** 32
    - **Validation:** Tracked validation loss to prevent overfitting
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with doc_col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Model Architecture")
    st.markdown("""
    A custom Convolutional Neural Network (CNN V1) inspired by VGG-style architectures:
    1. **Block 1:** Conv2D (32 filters, 3x3) + BatchNorm + MaxPooling2D (2x2)
    2. **Block 2:** Conv2D (64 filters, 3x3) + BatchNorm + MaxPooling2D (2x2)
    3. **Block 3:** Conv2D (128 filters, 3x3) + BatchNorm + MaxPooling2D (2x2)
    4. **Dense Head:** Flatten → Dense (128) → Dropout (0.1) → Dense (64) → Dropout (0.1)
    5. **Output:** Dense (1, Sigmoid Activation)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Explainability (Grad-CAM)")
    st.markdown("""
    To ensure the model is making decisions based on relevant features rather than spurious correlations, **Gradient-weighted Class Activation Mapping** is implemented. 
    It intercepts the gradients flowing into the final convolutional layer to generate a heatmap indicating the spatial areas most responsible for the model's prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### Future Improvements")
    st.markdown("""
    - Implement Transfer Learning using ResNet50V2 or EfficientNet for higher accuracy.
    - Expand data augmentation pipeline (rotations, zoom, flips) to increase robustness.
    - Containerize the application using Docker and deploy to AWS/GCP.
    - Set up a CI/CD pipeline using GitHub Actions for automated testing.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
