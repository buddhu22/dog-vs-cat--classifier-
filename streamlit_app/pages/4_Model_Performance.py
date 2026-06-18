import streamlit as st
import sys
import plotly.graph_objects as go
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.config import OUTPUTS_DIR, IMAGE_SIZE, TEST_DIR
from src.predict import Predictor
from src.evaluate import Evaluator
from streamlit_app.styles import inject_global_css, render_sidebar_branding

st.set_page_config(page_title="Performance", page_icon="", layout="wide")

inject_global_css()
render_sidebar_branding()

st.markdown('<div class="hero-title" style="background: linear-gradient(135deg, #f59e0b, #ef4444, #ec4899, #f59e0b); background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradientShift 4s ease infinite;">Model Performance Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Comprehensive evaluation metrics and visualizations based on the test dataset.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return Predictor()

col_btn, col_info = st.columns([1, 2])
with col_btn:
    st.markdown('<div style="padding-top: 1rem;">', unsafe_allow_html=True)
    run_eval = st.button("Run Evaluation on Test Data", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col_info:
    st.info("Running the evaluation performs a full forward pass on 5,000 test images and calculates the latest metrics.")

if run_eval:
    with st.spinner("Evaluating model... This may take a moment depending on the dataset size."):
        try:
            # Lazy-import TensorFlow only when the user clicks the button
            import tensorflow as tf
            from tensorflow.keras.utils import image_dataset_from_directory

            predictor = load_predictor()
            if predictor.model is None:
                st.error("Model could not be loaded. Please ensure `models/best_model.keras` exists.")
            else:
                evaluator = Evaluator()

                if not TEST_DIR.exists():
                    st.error(f"Test directory not found at {TEST_DIR}. Ensure you have test data.")
                else:
                    test_ds = image_dataset_from_directory(
                        str(TEST_DIR),
                        labels='inferred',
                        label_mode='int',
                        batch_size=32,
                        image_size=IMAGE_SIZE,
                        shuffle=False
                    )

                    y_true = []
                    y_prob = []

                    progress_text = "Operation in progress. Please wait."
                    my_bar = st.progress(0, text=progress_text)
                    
                    total_batches = len(test_ds)
                    
                    for batch_idx, (images, labels) in enumerate(test_ds):
                        # image_dataset_from_directory returns float32 in [0, 255]
                        # Normalize to [0, 1] for the model
                        images = images / 255.0
                        probs = predictor.model.predict(images, verbose=0)
                        
                        y_true.extend(labels.numpy().tolist())
                        y_prob.extend(probs[:, 0].tolist())
                        
                        my_bar.progress((batch_idx + 1) / total_batches, text=f"Processing batch {batch_idx+1}/{total_batches}")

                    # Convert probabilities to binary predictions (threshold 0.5)
                    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

                    metrics = evaluator.evaluate(y_true, y_pred, y_prob)
                    
                    my_bar.empty()
                    st.session_state['metrics'] = metrics
                    st.success("Evaluation complete!")
                    st.rerun()

        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ── Display metrics if available (either freshly generated or from previous run if state holds) ──
# For portfolio viewing, we'll try to read the classification_report.txt to show metrics even if not run in this session
metrics = None
if 'metrics' in st.session_state:
    metrics = st.session_state['metrics']
else:
    # Attempt to load from previous file
    report_path = OUTPUTS_DIR / "classification_report.txt"
    if report_path.exists():
        metrics = {}
        with open(report_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Accuracy:"): metrics['Accuracy'] = float(line.split(":")[1].strip())
                elif line.startswith("Precision:"): metrics['Precision'] = float(line.split(":")[1].strip())
                elif line.startswith("Recall:"): metrics['Recall'] = float(line.split(":")[1].strip())
                elif line.startswith("F1 Score:"): metrics['F1 Score'] = float(line.split(":")[1].strip())
        
        # If we failed to parse all 4, reset to None
        if len(metrics) < 4:
            metrics = None

if metrics:
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase;">Accuracy</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 2.2rem; font-weight: bold; background: linear-gradient(135deg, #10b981, #059669); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics["Accuracy"]:.2%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase;">Precision</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 2.2rem; font-weight: bold; background: linear-gradient(135deg, #3b82f6, #2563eb); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics["Precision"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase;">Recall</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 2.2rem; font-weight: bold; background: linear-gradient(135deg, #8b5cf6, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics["Recall"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="glass-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.9rem; color: #9ca3af; text-transform: uppercase;">F1 Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 2.2rem; font-weight: bold; background: linear-gradient(135deg, #f59e0b, #d97706); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{metrics["F1 Score"]:.4f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Plotly Dataset Distribution (hardcoded info for portfolio)
    st.markdown("### Dataset Distribution")
    fig = go.Figure(data=[go.Pie(
        labels=['Cats', 'Dogs'], 
        values=[12500, 12500], 
        hole=.4,
        marker=dict(colors=['#8b5cf6', '#f59e0b'])
    )])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Evaluation Plots")
    col_img1, col_img2, col_img3 = st.columns(3)
    
    def display_image(filename, title):
        path = OUTPUTS_DIR / filename
        if path.exists():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"##### {title}")
            st.image(str(path), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning(f"{filename} not found.")

    with col_img1: display_image("confusion_matrix.png", "Confusion Matrix")
    with col_img2: display_image("roc_curve.png", "ROC Curve")
    with col_img3: display_image("pr_curve.png", "Precision-Recall Curve")
        
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### Classification Report")
    report_path = OUTPUTS_DIR / "classification_report.txt"
    if report_path.exists():
        with open(report_path, 'r') as f:
            st.markdown(f'<div class="glass-card"><pre style="color: #d1d5db; background: transparent; border: none; font-size: 0.95rem;">{f.read()}</pre></div>', unsafe_allow_html=True)
    else:
        st.warning("Classification report not found.")
else:
    st.info("No metrics available. Click 'Run Evaluation on Test Data' to generate them.")
