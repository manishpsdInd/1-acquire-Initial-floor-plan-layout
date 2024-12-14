import cv2
import numpy as np
import os
from config import OUTPUT_DIR

def process_video(video_path, output_folder, max_frames=300):
    """
    Processes a video to extract layout features from individual frames.
    :param video_path: Path to the input video
    :param output_folder: Path to save the processed frames and aggregated results
    :param max_frames: Maximum number of frames to process (e.g., 300 for 10 seconds at 30 fps)
    :return: Aggregated layout data as a dictionary
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frame_count = 0
    layout_data = {"walls": []}

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray_frame, threshold1=100, threshold2=200)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                layout_data["walls"].append({"start": (x1, y1), "end": (x2, y2)})

        # Save the processed frame (optional)
        frame_output_path = os.path.join(output_folder, f"frame_{frame_count:03d}.png")
        cv2.imwrite(frame_output_path, edges)

        frame_count += 1

    cap.release()
    return layout_data
