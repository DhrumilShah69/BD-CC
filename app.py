import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model1.h5")
    return model

model = load_model()

# Set target image size (match your model input size)
TARGET_SIZE = (224, 224)

st.title("🫁 COPD Detection (Live Model)")
st.write("Upload a chest X-ray or CT scan image to detect COPD.")

uploaded_file = st.file_uploader("Choose a chest scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest Scan", use_column_width=True)
    st.write("🔍 Running model prediction...")

    # Preprocess the image
    img_resized = image.resize(TARGET_SIZE)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]  # Assuming binary classification with sigmoid output

    # Display result
    if prediction >= 0.5:
        st.error(f"🚨 COPD Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"✅ No COPD Detected (Confidence: {1 - prediction:.2f})")
