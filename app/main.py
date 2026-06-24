import json
import os

from PIL import Image

import numpy as np
import streamlit as st
import tensorflow as tf


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Rice Disease Scanner",
    page_icon="RD",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.05), transparent 26%),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.03), transparent 22%),
                linear-gradient(180deg, #232528 0%, #17191b 100%);
            color: #f1f1f1;
        }
        .stApp, .stApp p, .stApp span, .stApp div, .stApp li, .stApp label {
            color: #f1f1f1;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #ffffff;
        }
        .stButton button {
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: linear-gradient(180deg, #3b4045 0%, #2a2e33 100%);
            color: #ffffff;
            padding: 0.65rem 1rem;
            font-weight: 600;
        }
        .stButton button:hover {
            border-color: rgba(255, 255, 255, 0.22);
            background: linear-gradient(180deg, #454b51 0%, #32373d 100%);
            color: #ffffff;
        }
        .stFileUploader {
            background: rgba(28, 30, 33, 0.92);
            border-radius: 16px;
            padding: 0.25rem 0.5rem 0.75rem 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .hero-card {
            background: rgba(33, 36, 39, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(10px);
        }
        .metric-card {
            background: rgba(28, 30, 33, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.24);
        }
        .soft-label {
            color: #b7c0b9;
            font-size: 0.92rem;
            letter-spacing: 0.02em;
        }
        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #ffffff;
        }
        .result-subtitle {
            color: #c3c9c5;
            margin-bottom: 0;
        }
        .stCaption {
            color: #d0d3d1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================
@st.cache_resource
def load_model_and_data():
    """Load model and class indices (cached for performance)"""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    class_indices_path = os.path.join(working_dir, "class_indices.json")
    class_indices = json.load(open(class_indices_path))

    return model, class_indices


model, class_indices = load_model_and_data()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def load_and_preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    # Convert to RGB (removes alpha/transparency channel from PNG)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize the image
    img = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Keep raw pixel values because the model already contains its own rescaling layer.
    img_array = img_array.astype('float32')
    
    return img_array


def predict_image_class(model, image, class_indices):
    """Predict disease class and return prediction with confidence"""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence


# =====================================================
# SIDEBAR - INFORMATION
# =====================================================
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    to identify common **rice leaf diseases** from uploaded images.
    
    **Supported Classes:**
    - Bacterial blight
    - Blast
    - Brown spot
    - Tungro
    """)
    
    st.divider()
    
    st.header("Model Info")
    st.info("""
    - **Architecture:** CNN classifier
    - **Input Size:** 224x224 pixels
    - **Use Case:** Rice disease detection
    - **Output:** Disease label with confidence
    """)
    
    st.divider()
    
    st.header("Disclaimer")
    st.warning("""
    This app provides **diagnosis only**.
    It does not provide treatment or medicine advice.
    """)


# =====================================================
# MAIN APP
# =====================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="soft-label">Rice Leaf Disease Analysis</div>
        <h1 style="margin: 0.25rem 0 0.4rem 0;">Clean, fast and focused disease detection</h1>
        <p style="margin: 0; color: #50635a; font-size: 1.02rem; max-width: 760px;">
            Upload a rice leaf image to identify the predicted disease class and confidence score.
            The app is intentionally kept simple and does not show treatment suggestions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# File uploader
uploaded_image = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

# If image is uploaded
if uploaded_image is not None:
    # Display image
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns([1.05, 1])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Preview</div><p class="result-subtitle">Uploaded rice leaf image</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-title">Analysis</div><p class="result-subtitle">Prediction and confidence</p>', unsafe_allow_html=True)
        
        if st.button("Classify Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing image... Please wait..."):
                # Get prediction
                prediction, confidence = predict_image_class(model, image, class_indices)

                display_label = prediction.replace("_", " ")

                st.success(f"Predicted class: {display_label}")
                st.info(f"Confidence: {confidence:.2f}%")
                st.caption("Diagnosis only. No treatment advice is shown in this version.")

                st.markdown("### Quick Summary")
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.metric("Predicted label", display_label)
                with summary_col2:
                    st.metric("Confidence", f"{confidence:.2f}%")

                st.markdown("### What this app does")
                st.markdown(
                    """
                    - Detects the most likely rice disease class
                    - Shows a confidence score for the prediction
                    - Keeps the interface simple and focused
                    """
                )
        else:
            st.info("Upload an image and click **Classify Disease** to run the model.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information section
    st.divider()
    st.markdown("### Tips for Better Results")
    st.markdown(
        """
        - Use a clear, well-lit image of a single leaf
        - Keep the infected area visible in the frame
        - Avoid heavy blur, shadows, and strong glare
        - Prefer a close crop instead of a wide field photo
        """
    )

else:
    # Show placeholder when no image uploaded
    st.info("Upload a rice leaf image to start the analysis.")
    st.markdown("### Recommended image quality")
    st.markdown(
        """
        - Clear leaf details
        - Good lighting
        - Tight crop around the affected area
        - JPG, JPEG, or PNG format
        """
    )


# =====================================================
# FOOTER
# =====================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built by Yasir Fareed using Streamlit & TensorFlow | Plant Disease Classifier v1.0</p>
</div>

""", unsafe_allow_html=True)