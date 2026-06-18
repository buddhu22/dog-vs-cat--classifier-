import streamlit as st
import sys
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from streamlit_app.styles import inject_global_css, render_sidebar_branding

st.set_page_config(page_title="Project Overview", page_icon="", layout="wide")

inject_global_css()
render_sidebar_branding()

st.markdown('<div class="hero-title">Project Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'A deep dive into the architecture, objectives, and core modules of the Bio-Inspired Dog vs Cat Classifier.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Objective")
    st.markdown("""
    This project demonstrates the deployment of a fully trained Convolutional Neural Network (CNN) specifically tailored to differentiate between images of Dogs and Cats. 
    The goal was to build a robust, production-ready system incorporating not just inference, but also deep learning explainability and comprehensive evaluation metrics.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Modules Developed")
    st.markdown("""
    - **Prediction Pipeline (`predict.py`)**: Handles the loading of the `.keras` model, ensures consistent image preprocessing (256x256 RGB), and returns class probabilities.
    - **Explainability (`gradcam.py`)**: Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the regions of the input image that were most influential to the model's final decision.
    - **Evaluation (`evaluate.py`)**: A dedicated module for statistical analysis of the model's performance on test data, generating confusion matrices, ROC curves, and PR curves.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Architecture Stack")
    
    st.markdown("""
    <style>
    .tech-badge {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.5);
    }
    </style>
    <div style="margin-bottom: 15px;">
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">Keras</span>
        <span class="tech-badge">Python</span>
    </div>
    <div style="margin-bottom: 15px;">
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">HTML/CSS</span>
    </div>
    <div style="margin-bottom: 15px;">
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">Pillow</span>
    </div>
    <div style="margin-bottom: 15px;">
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Matplotlib</span>
        <span class="tech-badge">Seaborn</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### The CNN Model")
    st.markdown("""
    Convolutional Neural Networks are biologically inspired variants of multilayer perceptrons designed to emulate the behavior of the visual cortex. 
    By utilizing convolutional layers with learnable filters, the model extracts spatial hierarchies of features ranging from simple edges to complex shapes (like a cat's ear or a dog's snout).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.info("Explore the sidebar to test the model dynamically!")
