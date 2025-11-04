import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page title
st.title("🧠 Brain Tumor Segmentation App")
st.write("Upload an MRI image to predict the tumor area.")

# Load YOLOv8 model
model = YOLO("yolov8_model.pt")  # make sure your model is in same folder

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_file.name)
    image = Image.open(uploaded_file)
    image.save(img_path)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    st.write("🔍 Detecting tumor...")
    results = model.predict(source=img_path, save=True, conf=0.5)

    # Get result path
    result_path = results[0].save_dir
    output_files = os.listdir(result_path)

    # Display output image
    for file in output_files:
        if file.endswith(('.jpg', '.png')):
            st.image(os.path.join(result_path, file), caption="Predicted Output", use_container_width=True)
