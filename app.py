"""
Deepfake Image Detection System - Streamlit Web Application
Provides a professional, interactive interface for evaluating digital image authenticity
using deep convolutional neural networks with transfer learning (EfficientNetB0).
"""

import sys
import json
import io
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    DEFAULT_THRESHOLD,
    MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS,
    TARGET_SIZE,
    PREDICTION_DISCLAIMER,
)
from src.preprocessing import validate_image_file
from src.inference import DeepfakePredictor

# Page Setup
st.set_page_config(
    page_title="Deepfake Image Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphic Cards, Elegant Typography, Responsive Accents)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    line-height: 1.6;
    max-width: 850px;
}

/* Feature Badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.pill-active {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.pill-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Prediction Result Cards */
.result-card-real {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(6, 95, 70, 0.15) 100%);
    border: 2px solid #10B981;
    border-radius: 16px;
    padding: 1.75rem;
    margin: 1.5rem 0;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
}

.result-card-fake {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(153, 27, 27, 0.15) 100%);
    border: 2px solid #EF4444;
    border-radius: 16px;
    padding: 1.75rem;
    margin: 1.5rem 0;
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
}

.prediction-header {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem;
}

.prediction-real-text {
    color: #10B981;
}

.prediction-fake-text {
    color: #EF4444;
}

/* Metric Display Cards */
.score-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}

.score-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.score-label {
    font-size: 0.85rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Info Box */
.info-glass {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #CBD5E1;
}

.disclaimer-box {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin-top: 1.5rem;
    font-size: 0.85rem;
    color: #D1D5DB;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Caches the DeepfakePredictor instance to avoid re-loading weights on each rerun."""
    return DeepfakePredictor()


def render_sidebar(predictor: DeepfakePredictor) -> float:
    """Renders the sidebar with settings, model specs, and architecture guidance."""
    st.sidebar.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/lock/materialicons/48dp/2x/baseline_lock_white_48dp.png",
        width=48,
    )
    st.sidebar.markdown("### Model Configuration")

    # Decision Threshold Control
    threshold = st.sidebar.slider(
        "Decision Threshold (Fake Score)",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.get("global_threshold", DEFAULT_THRESHOLD)),
        step=0.05,
        key="global_threshold",
        help="If the fake-class score meets or exceeds this threshold, the system flags the image as LIKELY DEEPFAKE.",
    )

    if st.sidebar.button("🔄 Reload Model Weights", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Architecture")

    # Model Status
    if predictor.is_ready:
        st.sidebar.markdown(
            f'<div class="status-pill pill-active">● Model Active ({Path(str(predictor.model_path)).name})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-pill pill-warning">⚠ Model Offline (Train Required)</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        """
        - **Architecture:** EfficientNetB0
        - **Paradigm:** Transfer Learning (ImageNet)
        - **Input Res:** 224 × 224 × 3 (RGB)
        - **Classification:** Binary Sigmoid
        - **Class Map:** fake = 0, real = 1
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Verification Protocol")
    st.sidebar.caption(
        "Automated forensic analysis inspects subtle blending artifacts, frequency domain incongruities, "
        "and digital manipulation boundaries characteristic of face-swapping algorithms."
    )
    return float(threshold)


def render_home_page():
    """Renders the top overview section explaining deepfake detection mechanics."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="status-pill pill-active">CNN Forensic Analysis Engine</div>
            <div class="hero-title">Deepfake Image Detection System</div>
            <div class="hero-subtitle">
                Inspect digital portraits for algorithmic synthesis and facial manipulation using deep 
                Convolutional Neural Networks with transfer learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="info-glass">
                <strong>1. Feature Extraction</strong><br>
                EfficientNetB0 examines low-level pixel gradients, facial landmarks, and skin texture consistency.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        with st.container():
            st.markdown(
                """
                <div class="info-glass">
                    <strong>2. Boundary Artifacts</strong><br>
                    Deepfake models frequently leave microscopic boundary seams and warping around the eyes, jaw, and hair.
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col3:
        with st.container():
            st.markdown(
                """
                <div class="info-glass">
                    <strong>3. Score Evaluation</strong><br>
                    A calibrated sigmoid output produces dual-class prediction scores against a strict decision threshold.
                </div>
                """,
                unsafe_allow_html=True,
            )


def generate_report_text(result: dict, filename: str) -> str:
    """Creates a downloadable formatted text report."""
    return f"""=======================================================
DEEPFAKE IMAGE DETECTION AUDIT REPORT
=======================================================
Timestamp:             {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
Analyzed Filename:     {filename}
Final Prediction:      {result['prediction']}
Dominant Category:     {result['dominant_class'].upper()}
-------------------------------------------------------
CONFIDENCE SCORES
-------------------------------------------------------
Fake Class Score:      {result['fake_score']:.4f} ({result['fake_score_percent']}%)
Real Class Score:      {result['real_score']:.4f} ({result['real_score_percent']}%)
Decision Threshold:    {result['threshold']:.2f} ({result['threshold_percent']}%)
Overall Confidence:    {result['confidence_percent']}%
-------------------------------------------------------
TECHNICAL DETAILS
-------------------------------------------------------
Input Resolution:      {result.get('metadata', {}).get('original_width', 'N/A')}x{result.get('metadata', {}).get('original_height', 'N/A')} px
Target Processing:     224x224 RGB
Model Path:            {result.get('model_path', 'N/A')}
Explanation:
{result['explanation']}
-------------------------------------------------------
DISCLAIMER:
{result['disclaimer']}
======================================================="""


def main():
    predictor = load_predictor()
    selected_threshold = render_sidebar(predictor)
    render_home_page()

    st.markdown("### Image Analysis")
    st.write("Upload a portrait photograph to evaluate its authenticity. Supported formats: JPG, JPEG, PNG.")

    # Sensitivity Information
    with st.expander("⚙️ Decision Sensitivity & Detection Protocol", expanded=False):
        st.write(
            f"The model computes a **Fake Score** in [0%, 100%]. If the Fake Score meets or exceeds "
            f"the active decision threshold of **{selected_threshold * 100:.1f}%**, the image is classified as **LIKELY DEEPFAKE**.\n\n"
            f"You can adjust the decision threshold dynamically at any time using the slider in the left sidebar."
        )

    # Upload Control
    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=["jpg", "jpeg", "png"],
        help=f"Maximum file size allowed: {MAX_UPLOAD_SIZE_MB} MB",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        is_valid, val_msg, pil_img = validate_image_file(file_bytes)

        if not is_valid:
            st.error(f"⚠️ Validation Failed: {val_msg}")
            return

        col_img, col_metrics = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("##### Uploaded Image Preview")
            st.image(pil_img, use_column_width=True, caption=f"{uploaded_file.name} ({pil_img.width}×{pil_img.height} px)")
            st.caption(f"File Size: {len(file_bytes) / 1024:.1f} KB | Mode: {pil_img.mode}")

        with col_metrics:
            st.markdown("##### Forensic Analysis")

            if not predictor.is_ready:
                st.warning(
                    "⚠️ No trained model detected. Please ensure a model is trained using `python train.py`.\n\n"
                    "If you just started, generate benchmark data with `python prepare_dataset.py --generate_sample_data` "
                    "and train with `python train.py --epochs 3`."
                )
                return

            analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

            if analyze_btn:
                with st.spinner("Executing neural forward pass and feature extraction..."):
                    result = predictor.predict(pil_img, threshold=selected_threshold)

                if result["status"] == "error":
                    st.error(f"Inference Error: {result.get('error')}")
                    return

                # Render Results
                is_fake = result["prediction"] == "LIKELY DEEPFAKE"
                card_class = "result-card-fake" if is_fake else "result-card-real"
                text_class = "prediction-fake-text" if is_fake else "prediction-real-text"
                icon = "🚨" if is_fake else "✅"

                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="prediction-header {text_class}">{icon} {result['prediction']}</div>
                        <div style="color: #E2E8F0; font-size: 0.95rem;">{result['explanation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Dynamic Metric Cards: Fake Score, Real Score, and Overall Confidence
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-value" style="color: #EF4444;">{result['fake_score_percent']}%</div>
                            <div class="score-label">Fake Score</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sc2:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-value" style="color: #10B981;">{result['real_score_percent']}%</div>
                            <div class="score-label">Real Score</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with sc3:
                    st.markdown(
                        f"""
                        <div class="score-card">
                            <div class="score-value" style="color: #60A5FA;">{result['confidence_percent']}%</div>
                            <div class="score-label">Confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Contextual Decision Margin Info
                diff_from_thresh = result['fake_score_percent'] - result['threshold_percent']
                margin_str = f"+{diff_from_thresh:.2f}% above cutoff" if diff_from_thresh >= 0 else f"{abs(diff_from_thresh):.2f}% below cutoff"
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.5rem 1rem; margin: 0.8rem 0; font-size: 0.88rem; color: #94A3B8;">
                        🎯 <strong>Active Decision Boundary:</strong> Flags as DEEPFAKE when Fake Score ≥ {result['threshold_percent']}% &nbsp;|&nbsp; <strong>Margin:</strong> {margin_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Progress Bars for Visualization
                st.markdown("**Dual Class Score Distribution**")
                st.progress(float(result["fake_score"]), text=f"Synthetic / Manipulation Score: {result['fake_score_percent']}%")
                st.progress(float(result["real_score"]), text=f"Authenticity / Real Score: {result['real_score_percent']}%")

                # Export Options
                st.markdown("---")
                st.markdown("##### Export Detection Report")
                report_txt = generate_report_text(result, uploaded_file.name)
                report_json = json.dumps(result, indent=2)

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.download_button(
                        label="📄 Download Text Report (.txt)",
                        data=report_txt,
                        file_name=f"deepfake_report_{Path(uploaded_file.name).stem}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with dcol2:
                    st.download_button(
                        label="📊 Download JSON Report (.json)",
                        data=report_json,
                        file_name=f"deepfake_report_{Path(uploaded_file.name).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                # Mandatory Disclaimer
                st.markdown(
                    f"""
                    <div class="disclaimer-box">
                        <strong>Important Forensic Disclaimer:</strong><br>
                        {PREDICTION_DISCLAIMER}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
