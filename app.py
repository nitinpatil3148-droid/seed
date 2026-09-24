import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Seed Quality Assessment",
    page_icon="🌱",
    layout="centered"
)


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans"
]

SEED_NATURE = {
    "Broken soybeans": "Physically Damaged Seed",
    "Immature soybeans": "Immature / Developing Seed",
    "Intact soybeans": "Healthy / Intact Seed",
    "Skin-damaged soybeans": "Skin-Damaged Seed",
    "Spotted soybeans": "Spotted / Defected Seed"
}


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "soybean_quality_model_final.keras"
    )

    return model


model = load_model()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_seed(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image).astype("float32")

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Get highest probability class
    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index] * 100

    return (
        predicted_class,
        confidence,
        predictions
    )


# =========================================================
# FRONTEND
# =========================================================

st.title("🌱 AI-Based Seed Quality Assessment")

st.write(
    "Upload a soybean seed image or capture "
    "one using your device camera."
)

st.divider()


# =========================================================
# INPUT METHOD
# =========================================================

input_method = st.radio(
    "Select input method:",
    [
        "Upload Image",
        "Capture Image"
    ],
    horizontal=True
)


image = None


# =========================================================
# UPLOAD IMAGE
# =========================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a soybean seed image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


# =========================================================
# CAMERA
# =========================================================

else:

    camera_image = st.camera_input(
        "Take a picture of the soybean seed"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")


# =========================================================
# DISPLAY IMAGE
# =========================================================

if image is not None:

    st.subheader("Seed Image")

    st.image(
        image,
        width=350
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

if image is not None:

    if st.button(
        "🔍 Analyze Seed",
        type="primary"
    ):

        with st.spinner(
            "Analyzing seed..."
        ):

            (
                predicted_class,
                confidence,
                predictions
            ) = predict_seed(image)


        st.divider()

        st.subheader(
            "🌱 Classification Result"
        )

        st.success(
            f"Predicted Class: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # =================================================
        # ALL PROBABILITIES
        # =================================================

        st.subheader(
            "Prediction Probabilities"
        )

        for class_name, probability in zip(
            CLASS_NAMES,
            predictions
        ):

            percentage = probability * 100

            st.write(
                f"**{class_name}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                float(probability)
            )

        st.divider()
        st.subheader("🌱 Final Seed Quality Assessment")

        seed_nature = SEED_NATURE[predicted_class]

        st.success(f"Seed Nature: {seed_nature}")
