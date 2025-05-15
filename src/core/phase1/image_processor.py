import cv2
import numpy as np
import os

def process_floorplan_image(image_path, output_path):
    """
    Processes an image-based floor plan to detect key layout features.
    :param image_path: Path to the floor plan image
    :param output_path: Path to save the processed visualization
    :return: Parsed layout as a dictionary
    """
    layout_data = {"walls": []}

    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}. Ensure it's a valid image file.")

    # Apply edge detection
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)
    if lines is not None:
        for line in lines:
            # Convert NumPy data types to Python native types
            x1, y1, x2, y2 = map(int, line[0])
            layout_data["walls"].append({"start": (x1, y1), "end": (x2, y2)})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save visualization of edges
    cv2.imwrite(output_path, edges)
    return layout_data
