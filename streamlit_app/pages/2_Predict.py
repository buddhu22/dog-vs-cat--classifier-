import streamlit as st
import tempfile
import pandas as pd
import plotly.graph_objects as go
import sys
from PIL import Image
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.predict import Predictor
from streamlit_app.styles import inject_global_css, render_sidebar_branding

st.set_page_config(page_title="Predict", page_icon="", layout="wide")

# Inject global CSS and sidebar
inject_global_css()
render_sidebar_branding()

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title hero-title-predict">Upload & Predict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Drop an image of a Dog or a Cat and watch the CNN classify it in real-time with confidence scores.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return Predictor()

predictor = load_predictor()

# ── File uploader ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
        # Show image metadata
        w, h = image.size
        st.caption(f"{w}×{h} px  •  {uploaded_file.name}  •  {uploaded_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Prediction Results")

        if predictor.model is None:
            st.error(
                f"Model could not be loaded. Please check if `{predictor.model_path}` exists."
            )
        else:
            with st.spinner("Analyzing image..."):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)

                    predicted_class, confidence = predictor.predict_single(tmp_path)

                    # ── Prediction Badge ──
                    icon = "" if predicted_class == "Dog" else ""
                    badge_class = "badge-dog" if predicted_class == "Dog" else "badge-cat"
                    accent = "#f59e0b" if predicted_class == "Dog" else "#8b5cf6"

                    st.markdown(
                        f'<div class="prediction-badge {badge_class}">'
                        f'{icon} {predicted_class}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Confidence Meter ──
                    st.markdown(
                        f"""
                        <div class="confidence-container">
                            <div class="confidence-label">CONFIDENCE</div>
                            <div class="confidence-value" style="color:{accent}">{confidence:.1f}%</div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" style="width:{confidence}%; background:linear-gradient(90deg, {accent}, {accent}dd);"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                    # ── Probability Chart (Plotly) ──
                    st.markdown("#### Probability Distribution")
                    cat_prob = confidence if predicted_class == "Cat" else 100 - confidence
                    dog_prob = confidence if predicted_class == "Dog" else 100 - confidence

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Cat", "Dog"],
                        y=[cat_prob, dog_prob],
                        marker=dict(
                            color=["#8b5cf6", "#f59e0b"],
                            line=dict(width=0),
                        ),
                        text=[f"{cat_prob:.1f}%", f"{dog_prob:.1f}%"],
                        textposition="outside",
                        textfont=dict(size=14, color="#e5e7eb"),
                        width=0.5,
                    ))
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(
                            range=[0, 110],
                            showgrid=True,
                            gridcolor="rgba(255,255,255,0.05)",
                            title="Probability (%)",
                            title_font=dict(color="#9ca3af"),
                            tickfont=dict(color="#9ca3af"),
                        ),
                        xaxis=dict(
                            tickfont=dict(size=14, color="#e5e7eb"),
                        ),
                        height=280,
                        margin=dict(l=40, r=20, t=20, b=40),
                        bargap=0.4,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Download button ──
                    result_text = (
                        f"═══════════════════════════════════\n"
                        f"  CNN DOG vs CAT — PREDICTION REPORT\n"
                        f"═══════════════════════════════════\n\n"
                        f"  File:       {uploaded_file.name}\n"
                        f"  Prediction: {predicted_class}\n"
                        f"  Confidence: {confidence:.2f}%\n"
                        f"  Cat Prob:   {cat_prob:.2f}%\n"
                        f"  Dog Prob:   {dog_prob:.2f}%\n\n"
                        f"═══════════════════════════════════\n"
                    )
                    st.download_button(
                        label="Download Report",
                        data=result_text,
                        file_name="prediction_result.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()

        st.markdown('</div>', unsafe_allow_html=True)

else:
    # ── Empty state with animated hint ──
    st.markdown("")
    st.markdown(
        """
        <div class="upload-hint">
            <div class="upload-icon"></div>
            <div class="upload-hint-text">
                Drag & drop an image above or click <strong>Browse files</strong><br>
                to see the CNN model classify it instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
