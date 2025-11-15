import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load your trained model
model = tf.keras.models.load_model("img_reco.h5")  # replace with your model file

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) to get the prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((28, 28))  # resize to 28x28 pixels
    image = ImageOps.invert(image)   # invert colors if needed
    img_array = np.array(image) / 255.0  # normalize

    img_array = img_array.reshape(1, 28*28)  # Flatten to shape (1, 784)

    img_array = img_array.reshape(1, 28, 28, 1)  # reshape for model

    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"Predicted Digit: **{predicted_digit}**")

