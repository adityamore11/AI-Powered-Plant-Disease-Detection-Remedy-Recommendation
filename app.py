"""Streamlit interface for local Plant Doctor AI inference."""
from __future__ import annotations
from io import BytesIO
import streamlit as st
from PIL import Image
from config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, MODEL_PATH
from src.predict import create_gradcam_overlay, generate_gradcam, load_model, predict_disease
from src.remedy import get_disease_info
from src.utils import load_class_names

st.set_page_config(page_title="Plant Doctor AI", page_icon="🌱", layout="wide")
st.title("🌱 Plant Doctor AI")
st.caption("AI-powered plant disease detection and remedy recommendation")

@st.cache_resource(show_spinner=False)
def cached_model(): return load_model(MODEL_PATH)

def bullets(items):
    for item in items: st.markdown(f"- {item}")

with st.sidebar:
    st.header("About")
    st.write("A local AI-assisted screening tool built with EfficientNetB0 and Streamlit.")
    st.write("**Model:** EfficientNetB0 transfer learning")
    st.write(f"**Confidence threshold:** {CONFIDENCE_THRESHOLD:.0%}")
    if CLASS_NAMES_PATH.exists(): st.write(f"**Supported classes:** {len(load_class_names(CLASS_NAMES_PATH))}")
    else: st.write("**Supported classes:** available after training")
    st.divider(); st.caption("Not a substitute for a professional agricultural diagnosis. Follow local agricultural guidance and product labels.")

uploaded = st.file_uploader("Upload a new leaf image", type=["jpg", "jpeg", "png"], help="Use a clear, well-lit image showing most of one leaf.")
if uploaded:
    uploaded_bytes = uploaded.getvalue()
    upload_key = hash(uploaded_bytes)
    if st.session_state.get("upload_key") != upload_key:
        st.session_state.upload_key = upload_key
        st.session_state.pop("prediction", None)
        st.session_state.pop("prediction_image", None)
        st.session_state.show_gradcam = False
    try:
        image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
        st.image(image, caption="Original leaf image", width=420)
    except Exception:
        st.error("That file could not be read as an image. Please upload a valid JPG, JPEG, or PNG."); st.stop()

if st.button("Detect Disease", type="primary", disabled=uploaded is None):
    try:
        with st.spinner("Analyzing the leaf image..."):
            model = cached_model(); prediction = predict_disease(image, model=model)
        st.session_state.prediction = prediction
        st.session_state.prediction_image = uploaded_bytes
    except FileNotFoundError as exc: st.error(str(exc))
    except Exception as exc: st.error(f"Prediction could not be completed: {exc}")
elif uploaded is None:
    st.info("Upload a JPG, JPEG, or PNG leaf image, then select Detect Disease.")

# Render from session state so interactions such as the Grad-CAM checkbox do not
# hide the last result on Streamlit's automatic rerun.
if st.session_state.get("prediction"):
    prediction = st.session_state.prediction
    image = Image.open(BytesIO(st.session_state.prediction_image)).convert("RGB")
    if prediction["is_confident"]:
        st.success("Detection completed")
    else:
        st.warning("The model is not sufficiently confident. Upload a clearer, well-lit image showing the full leaf, or consult an agricultural expert if symptoms persist.")
    st.subheader("Detection result")
    left, middle, right = st.columns(3); left.metric("Plant", prediction["plant"]); middle.metric("Disease", prediction["disease"]); right.metric("Confidence", f"{prediction['confidence']:.1%}")
    info = get_disease_info(prediction["class_name"])
    if "healthy" in prediction["class_name"].lower():
        st.success("✅ Plant appears healthy")
        st.write("Continue regular monitoring, appropriate watering, sanitation, and crop-specific preventive care.")
    else:
        st.subheader("Symptoms"); bullets(info.get("symptoms", []))
        st.subheader("Recommended actions"); bullets(info.get("recommended_actions", []))
        st.subheader("Prevention"); bullets(info.get("prevention", []))
        st.subheader("Precautions"); bullets(info.get("precautions", []))
        if info.get("sources"):
            st.subheader("Further reading")
            for source in info["sources"]:
                st.markdown(f"- [{source['label']}]({source['url']})")
    with st.expander("Top predictions"):
        for index, item in enumerate(prediction["top_predictions"], 1): st.write(f"{index}. {item['plant']} — {item['disease']}: **{item['confidence']:.1%}**")
    if st.checkbox("Show Grad-CAM explanation", key="show_gradcam"):
        try:
            model = cached_model()
            heatmap = generate_gradcam(model, image)
            overlay = create_gradcam_overlay(image, heatmap)
            st.image(overlay, caption="Grad-CAM overlay — highlighted regions contributed strongly to the prediction.", width=420)
        except Exception as exc: st.info(f"Grad-CAM is unavailable for this model: {exc}")
