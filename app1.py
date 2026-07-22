import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Fish Image Classification",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Image Classification")
st.write("Upload a fish image to predict its species.")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("best_fish_classifier_patched.h5", compile=False)

model = load_my_model()

# -------------------------------
# Load Class Names
# -------------------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224,224))

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a Fish Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        img = preprocess_image(image)

        prediction = model.predict(img)

        predicted_index = np.argmax(prediction)

        confidence = prediction[0][predicted_index]

        predicted_class = index_to_class[predicted_index]
        display_name = predicted_class.replace("_", " ").title()

        st.success(f"Prediction : {display_name}")

        st.info(f"Confidence : {confidence*100:.2f}%")

        st.subheader("Top 3 Predictions")

        sorted_indices = np.argsort(prediction[0])[::-1]

        for idx in sorted_indices[:3]:

            st.write(
                f"**{index_to_class[idx]}** : {prediction[0][idx]*100:.2f}%"
            )

            st.progress(float(prediction[0][idx]))