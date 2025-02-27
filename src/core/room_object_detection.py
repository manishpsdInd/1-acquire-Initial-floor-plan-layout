import torch
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import torch
import config


# Define paths
input_image_path = os.path.join(config.INPUT_DIR, "room.jpg")
#input_video_path = os.path.join(config.INPUT_DIR, "Video_2024-12-17.mp4")
#input_video_path = os.path.join(config.INPUT_DIR, "Video_2024-01-11.mp4")
#input_video_path = os.path.join(config.INPUT_DIR, "Video_2024-02-26.mp4")
#input_video_path = os.path.join(config.INPUT_DIR, "Video_2025-02-24.mp4")
input_video_path = os.path.join(config.INPUT_DIR, "Video_2025-02-25.mp4")


def from_image():
    # Load the YOLO model (pretrained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Load and display the image
    img = Image.open(input_image_path)

    # Perform object detection
    results = model(img)

    # Show results
    results.show()  # Display image with bounding boxes

    # Print detected objects
    results.print()  # Print object labels and confidence scores


def from_video_old():
    import torch
    import cv2

    # Load the YOLO model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open the video file (or use 0 for webcam)
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left

        # Perform object detection on the current frame
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results.render()[0]

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', annotated_frame)

        # Press 'q' to quit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


# =========================
# HYPERPARAMETER SETTINGS
MODEL_NAME = 'yolov5s'  # Change to 'yolov5m', 'yolov5l', or 'yolov5x' for better accuracy
CONF_THRESHOLD = 0.3  # Confidence threshold for object detection (increase to reduce false positives)
IOU_THRESHOLD = 0.5  # IoU threshold for non-max suppression (lower removes overlapping boxes)
IMAGE_SIZE = 640  # Input image size (increase for better accuracy, but slower inference)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available

# =========================
# LOAD YOLO MODEL (FINE-TUNED)
# =========================
model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True)
model.conf = CONF_THRESHOLD  # Apply custom confidence threshold
model.iou = IOU_THRESHOLD  # Apply custom IoU threshold
model.imgsz = IMAGE_SIZE  # Set input image size
model.to(DEVICE)  # Move model to GPU if available

# =========================
# VIDEO PROCESSING
# =========================
def from_video_hyperparameter():
    cap = cv2.VideoCapture(input_video_path)

    # Check if the video is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are left

        # Perform object detection on the current frame
        results = model(frame)

        # Render the results on the frame
        annotated_frame = results.render()[0]

        # Display the frame with bounding boxes
        cv2.imshow('Object Detection', annotated_frame)

        # Press 'q' to quit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()



#from_image()
#from_video()
from_video_hyperparameter()

