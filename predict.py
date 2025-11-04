from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("yolov8_model.pt")

# Run prediction on test images
results = model.predict(source="test_images", save=True, conf=0.5)

print("✅ Prediction complete! Check the 'runs/segment/predict' folder for results.")
