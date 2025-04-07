from ultralytics import YOLO

MODEL_NAME = "yolov8m.pt"

# Call YOLO just to download the model while connected to the internet
YOLO(MODEL_NAME)
