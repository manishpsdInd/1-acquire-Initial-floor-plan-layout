import cv2
import numpy as np

def process_floorplan_image(image_path, output_path):
    """
    Processes an image-based floor plan to detect key layout features.
    :param image_path: Path to the floor plan image
    :param output_path: Path to save the processed visualization
    :return: Parsed layout as a dictionary
    """
    layout_data = {"walls": [], "doors": []}
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Edge detection to identify walls
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            layout_data["walls"].append({"start": (x1, y1), "end": (x2, y2)})
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save visualization
    cv2.imwrite(output_path, image)

    return layout_data
