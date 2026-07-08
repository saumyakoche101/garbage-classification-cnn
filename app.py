
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("garbage_classifier.h5")

# Class names
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="Garbage Classification", page_icon="♻️")

st.title("♻️ Smart Garbage Classification System")

st.markdown("### AI-Powered Waste Segregation using Convolutional Neural Networks")

st.write("Upload an image to predict the garbage category.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image")

    img = img.resize((150,150))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(f"♻️ Predicted Category: {predicted_class.capitalize()}")
    
st.info(f"🎯 Confidence: {confidence:.2f}%")
