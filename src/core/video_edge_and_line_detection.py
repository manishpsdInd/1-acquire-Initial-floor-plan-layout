import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import config


# Define paths
input_video_path = os.path.join(config.INPUT_DIR, "Video_2024-12-17.mp4")
output_frames_path = os.path.join(config.OUTPUT_DIR, "extracted_frames_II")
os.makedirs(output_frames_path, exist_ok=True)

# Extract frames from video
def extract_frames(video_path, output_folder, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Unable to open video file."

    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends

        # Save every nth frame (controlled by frame_skip)
        if frame_count % frame_skip == 0:
            output_file = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(output_file, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return f"Frames extracted: {saved_count}"


# Load and display a sample frame for edge detection
sample_frame_path = os.path.join(output_frames_path, "frame_0000.png")

def display_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=50, threshold2=150)  # Apply edge detection

    # Display original and edge-detected images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Frame")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Edge Detection")
    plt.tight_layout()
    plt.show()


def detect_and_draw_lines(image_path):
    """
    Detects edges and straight lines using the Hough Transform.
    Overlays the detected lines on the original image.
    """
    # Read the image in grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # Draw detected lines on the original image
    img_with_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines

    # Display results: edges and lines overlaid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title("Edge Detection")
    axes[1].imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Hough Line Detection")
    plt.tight_layout()
    plt.show()


def filter_and_draw_lines(image_path, length_threshold=100, angle_tolerance=10):
    """
    Detects edges, filters prominent lines based on length and orientation,
    and overlays the filtered lines on the original image.
    """

    def line_length(x1, y1, x2, y2):
        """Calculate the Euclidean length of a line."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(x1, y1, x2, y2):
        """Calculate the angle of the line in degrees."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    # Read the image in grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # Filter lines based on length and angle
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = line_length(x1, y1, x2, y2)
            angle = line_angle(x1, y1, x2, y2)
            # Keep lines that are long enough and close to horizontal/vertical orientation
            if length > length_threshold and (abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance):
                filtered_lines.append((x1, y1, x2, y2))

    # Draw filtered lines on the original image
    img_with_filtered_lines = img.copy()
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(img_with_filtered_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines for filtered results

    # Display results: edges and filtered lines overlaid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(edges, cmap="gray")
    axes[0].set_title("Edge Detection")
    axes[1].imshow(cv2.cvtColor(img_with_filtered_lines, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Filtered Hough Lines")
    plt.tight_layout()
    plt.show()

#
# # Load pre-trained object detection model (Faster R-CNN for occlusion detection)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'fasterrcnn_resnet50_fpn', pretrained=True)
# model.eval()
#
# def detect_occlusions_and_refine_lines(image_path, length_threshold=100, angle_tolerance=10):
#     """
#     Detect occlusions (objects), mask occluded areas, and refine edge/line detection.
#     """
#
#     def line_length(x1, y1, x2, y2):
#         return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#
#     def line_angle(x1, y1, x2, y2):
#         return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
#
#     # Load the image
#     img = cv2.imread(image_path)
#     pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     # Convert the image for object detection
#     transform = T.ToTensor()
#     img_tensor = transform(pil_image)
#     predictions = model([img_tensor])
#
#     # Create an occlusion mask
#     occlusion_mask = np.zeros_like(img[:, :, 0])
#     for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
#         if score > 0.6:  # Confidence threshold
#             x1, y1, x2, y2 = map(int, box.tolist())
#             cv2.rectangle(occlusion_mask, (x1, y1), (x2, y2), 255, thickness=-1)
#
#     # Mask the image to ignore occlusions
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_masked = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(occlusion_mask))
#     edges = cv2.Canny(gray_masked, threshold1=50, threshold2=150)
#
#     # Apply Hough Transform for line detection
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
#
#     # Filter lines
#     filtered_lines = []
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             length = line_length(x1, y1, x2, y2)
#             angle = line_angle(x1, y1, x2, y2)
#             if length > length_threshold and (abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance):
#                 filtered_lines.append((x1, y1, x2, y2))
#
#     # Draw results
#     img_with_filtered_lines = img.copy()
#     for x1, y1, x2, y2 in filtered_lines:
#         cv2.line(img_with_filtered_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # Display results
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     axes[0].imshow(occlusion_mask, cmap="gray")
#     axes[0].set_title("Detected Occlusions (Mask)")
#     axes[1].imshow(edges, cmap="gray")
#     axes[1].set_title("Edge Detection (Masked)")
#     axes[2].imshow(cv2.cvtColor(img_with_filtered_lines, cv2.COLOR_BGR2RGB))
#     axes[2].set_title("Filtered Hough Lines (Refined)")
#     plt.tight_layout()
#     plt.show()


def improved_hough_lines_after_occlusion(image_path, length_threshold=120, max_line_gap=15):
    """
    Apply occlusion masking and re-run Hough Line Transform with improved parameters.
    """

    def line_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect objects (occlusions)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours and create an occlusion mask
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occlusion_mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(occlusion_mask, [contour], -1, 255, thickness=-1)

    # Mask the image to remove occluded areas
    gray_masked = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(occlusion_mask))
    edges = cv2.Canny(gray_masked, threshold1=50, threshold2=150)

    # Apply Hough Transform for line detection with improved parameters
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=length_threshold,
                            maxLineGap=max_line_gap)

    # Draw refined Hough Lines
    img_with_improved_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_with_improved_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines for improved output

    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(thresh_cleaned, cmap="gray")
    axes[0].set_title("Occlusion Detection (Contours)")
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Edge Detection (Masked)")
    axes[2].imshow(cv2.cvtColor(img_with_improved_lines, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Improved Hough Lines")
    plt.tight_layout()
    plt.show()


def merge_nearby_lines(lines, distance_threshold=20, angle_threshold=5):
    """
    Merge lines that are close in position and orientation.
    """
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        merged = False

        # Compare with already-merged lines
        for merged_line in merged_lines:
            mx1, my1, mx2, my2, mangle = merged_line
            dist = np.sqrt((x1 - mx1) ** 2 + (y1 - my1) ** 2)
            if abs(angle - mangle) < angle_threshold and dist < distance_threshold:
                # Merge lines: average positions
                mx1 = (mx1 + x1) // 2
                my1 = (my1 + y1) // 2
                mx2 = (mx2 + x2) // 2
                my2 = (my2 + y2) // 2
                merged_lines.remove(merged_line)
                merged_lines.append((mx1, my1, mx2, my2, angle))
                merged = True
                break

        if not merged:
            merged_lines.append((x1, y1, x2, y2, angle))

    return [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in merged_lines]


from collections import Counter


def filter_dominant_angles(lines, angle_tolerance=5):
    """
    Keep only lines aligned with dominant orientations (e.g., horizontal and vertical).
    """
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        angles.append(int(angle))

    # Find dominant angles
    angle_counter = Counter(angles)
    dominant_angles = [angle for angle, count in angle_counter.most_common(2)]

    # Filter lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if any(abs(angle - d_angle) < angle_tolerance for d_angle in dominant_angles):
            filtered_lines.append(line)
    return filtered_lines


def refine_hough_lines_pipeline(image_path, length_threshold=120, distance_threshold=20, angle_tolerance=5):
    """
    Full refinement pipeline for Hough Lines:
    - Apply occlusion masking
    - Detect edges using Canny
    - Run Hough Transform
    - Merge redundant lines
    - Filter by length and dominant angles
    """
    def line_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def merge_nearby_lines(lines, distance_threshold, angle_threshold):
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(x1, y1, x2, y2)
            merged = False

            for merged_line in merged_lines:
                mx1, my1, mx2, my2, mangle = merged_line
                dist = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)
                if abs(angle - mangle) < angle_threshold and dist < distance_threshold:
                    mx1 = (mx1 + x1) // 2
                    my1 = (my1 + y1) // 2
                    mx2 = (mx2 + x2) // 2
                    my2 = (my2 + y2) // 2
                    merged_lines.remove(merged_line)
                    merged_lines.append((mx1, my1, mx2, my2, angle))
                    merged = True
                    break

            if not merged:
                merged_lines.append((x1, y1, x2, y2, angle))
        return [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in merged_lines]

    # Load the image and apply occlusion masking
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occlusion_mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(occlusion_mask, [contour], -1, 255, thickness=-1)
    gray_masked = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(occlusion_mask))
    edges = cv2.Canny(gray_masked, threshold1=50, threshold2=150)

    # Apply Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # Merge and filter lines
    if lines is not None:
        lines = merge_nearby_lines(lines, distance_threshold, angle_tolerance)
        filtered_lines = []
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            length = line_length(x1, y1, x2, y2)
            angle = line_angle(x1, y1, x2, y2)
            if length > length_threshold:
                filtered_lines.append((x1, y1, x2, y2))
                angles.append(angle)

        # Draw final refined lines
        img_with_refined_lines = img.copy()
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(img_with_refined_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(thresh_cleaned, cmap="gray")
        axes[0].set_title("Occlusion Detection (Contours)")
        axes[1].imshow(edges, cmap="gray")
        axes[1].set_title("Edge Detection (Masked)")
        axes[2].imshow(cv2.cvtColor(img_with_refined_lines, cv2.COLOR_BGR2RGB))
        axes[2].set_title("Refined Hough Lines")
        plt.tight_layout()
        plt.show()


# Full pipeline for refining and drawing Hough Lines
def refine_and_draw_hough_lines(image_path, length_threshold=100, distance_threshold=20, angle_tolerance=5):
    """
    Full refinement pipeline for Hough Lines:
    - Detect edges using Canny
    - Run Hough Transform
    - Merge redundant lines
    - Filter by length and angles
    - Draw the detected walls in green
    """
    def line_length(x1, y1, x2, y2):
        """Calculate the Euclidean length of a line."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(x1, y1, x2, y2):
        """Calculate the angle of the line in degrees."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def merge_nearby_lines(lines, distance_threshold, angle_threshold):
        """Merge lines that are close in position and orientation."""
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(x1, y1, x2, y2)
            merged = False
            for merged_line in merged_lines:
                mx1, my1, mx2, my2, mangle = merged_line
                dist = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)
                if abs(angle - mangle) < angle_threshold and dist < distance_threshold:
                    mx1 = (mx1 + x1) // 2
                    my1 = (my1 + y1) // 2
                    mx2 = (mx2 + x2) // 2
                    my2 = (my2 + y2) // 2
                    merged_lines.remove(merged_line)
                    merged_lines.append((mx1, my1, mx2, my2, angle))
                    merged = True
                    break
            if not merged:
                merged_lines.append((x1, y1, x2, y2, angle))
        return [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in merged_lines]

    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Apply Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Merge and filter lines
    filtered_lines = []
    if lines is not None:
        lines = merge_nearby_lines(lines, distance_threshold, angle_tolerance)
        for line in lines:
            x1, y1, x2, y2 = line
            length = line_length(x1, y1, x2, y2)
            if length > length_threshold:
                filtered_lines.append((x1, y1, x2, y2))

    # Draw final detected walls in green
    img_with_walls = img.copy()
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(img_with_walls, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green lines

    # Save and return the image
    #output_path = "/data/output/refined_walls_output.png"
    output_path = os.path.join(config.OUTPUT_DIR, "refined_walls_output.png")
    cv2.imwrite(output_path, img_with_walls)

    img = cv2.imread(output_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title("Refined Walls Detected in Green")
    plt.axis("off")
    plt.show()


# Full pipeline for refining and drawing Hough Lines
def refine_and_draw_hough_lines(image_path, length_threshold=100, distance_threshold=20, angle_tolerance=5):
    """
    Full refinement pipeline for Hough Lines:
    - Detect edges using Canny
    - Run Hough Transform
    - Merge redundant lines
    - Filter by length and angles
    - Draw the detected walls in green
    """
    def line_length(x1, y1, x2, y2):
        """Calculate the Euclidean length of a line."""
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(x1, y1, x2, y2):
        """Calculate the angle of the line in degrees."""
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def merge_nearby_lines(lines, distance_threshold, angle_threshold):
        """Merge lines that are close in position and orientation."""
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = line_angle(x1, y1, x2, y2)
            merged = False
            for merged_line in merged_lines:
                mx1, my1, mx2, my2, mangle = merged_line
                dist = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)
                if abs(angle - mangle) < angle_threshold and dist < distance_threshold:
                    mx1 = (mx1 + x1) // 2
                    my1 = (my1 + y1) // 2
                    mx2 = (mx2 + x2) // 2
                    my2 = (my2 + y2) // 2
                    merged_lines.remove(merged_line)
                    merged_lines.append((mx1, my1, mx2, my2, angle))
                    merged = True
                    break
            if not merged:
                merged_lines.append((x1, y1, x2, y2, angle))
        return [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in merged_lines]

    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Apply Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Merge and filter lines
    filtered_lines = []
    if lines is not None:
        lines = merge_nearby_lines(lines, distance_threshold, angle_tolerance)
        for line in lines:
            x1, y1, x2, y2 = line
            length = line_length(x1, y1, x2, y2)
            if length > length_threshold:
                filtered_lines.append((x1, y1, x2, y2))

    # Draw final detected walls in green
    img_with_walls = img.copy()
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(img_with_walls, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Thicker green lines

        # Save and return the image
        output_path = os.path.join(config.OUTPUT_DIR, "refine_and_draw_hough_lines.png")
        cv2.imwrite(output_path, img_with_walls)

        img = cv2.imread(output_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        plt.title("Refine and Draw Hough Lines")
        plt.axis("off")
        plt.show()


# Fix the line merging logic
def aggregate_hough_lines_across_frames_fixed(frame_folder, length_threshold=100, distance_threshold=20, angle_tolerance=5):
    """
    Aggregate Hough Lines across multiple frames to detect consistent structures.
    """
    aggregated_lines = []

    def line_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def line_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    def merge_lines(lines, distance_threshold, angle_threshold):
        """Merge lines that are similar across frames."""
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = line_angle(x1, y1, x2, y2)
            merged = False
            for i, (mx1, my1, mx2, my2, mangle) in enumerate(merged_lines):
                dist = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)
                if abs(angle - mangle) < angle_threshold and dist < distance_threshold:
                    # Update the line in-place
                    merged_lines[i] = (
                        (mx1 + x1) // 2,
                        (my1 + y1) // 2,
                        (mx2 + x2) // 2,
                        (my2 + y2) // 2,
                        (mangle + angle) / 2,
                    )
                    merged = True
                    break
            if not merged:
                merged_lines.append((x1, y1, x2, y2, angle))
        return [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in merged_lines]

    # Process all frames in the folder
    for frame_file in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame_file)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, threshold1=50, threshold2=150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = line_length(x1, y1, x2, y2)
                if length > length_threshold:
                    aggregated_lines.append((x1, y1, x2, y2))

    # Merge lines across frames
    final_lines = merge_lines(aggregated_lines, distance_threshold, angle_tolerance)
    return final_lines


# Redefine the visualization function
def visualize_aggregated_lines(image_path, lines):
    """
    Visualize aggregated lines on a given image.
    """
    img = cv2.imread(image_path)
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw lines in green
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Aggregated Lines Across Frames")
    plt.axis("off")
    plt.show()


# Load the MiDaS model for depth estimation
# midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
# midas_model.eval()

# Define the transformation pipeline for the MiDaS model
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Use the appropriate transformation based on the model type
# transform = midas_transforms.default_transform

# Directory for saving depth maps
# depth_maps_path = "/mnt/data/depth_maps"
# os.makedirs(depth_maps_path, exist_ok=True)

# Function to estimate depth for a single frame
# def estimate_depth(image_path, output_folder):
#     """
#     Estimate depth for a single frame and save the depth map.
#     """
#     img = Image.open(image_path).convert("RGB")
#     input_batch = transform(img).unsqueeze(0)
#
#     with torch.no_grad():
#         prediction = midas_model(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.size[::-1],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#         depth_map = prediction.numpy()
#
#     # Save the depth map as an image
#     depth_map_path = os.path.join(output_folder, os.path.basename(image_path))
#     depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
#     depth_map_image = (depth_map_normalized * 255).astype("uint8")
#     cv2.imwrite(depth_map_path, depth_map_image)
#
#     return depth_map_path
#



# Extract frames
#extract_frames(input_video_path, output_frames_path, frame_skip=5)

# Display the first frame with edge detection applied
#display_image(sample_frame_path)

# Process the first frame with Hough Line Transform
#detect_and_draw_lines(sample_frame_path)

# Process the first frame with filtered line detection
#filter_and_draw_lines(sample_frame_path, length_threshold=100, angle_tolerance=10)

# Run the occlusion-aware pipeline on the sample frame
#detect_occlusions_and_refine_lines(sample_frame_path, length_threshold=100, angle_tolerance=10)

# Apply the improved Hough Line detection
#improved_hough_lines_after_occlusion(sample_frame_path, length_threshold=120, max_line_gap=15)

# Run the full pipeline on the sample frame
#refine_hough_lines_pipeline(sample_frame_path, length_threshold=120, distance_threshold=20, angle_tolerance=5)

# Display the refined image with detected walls in green
#refine_and_draw_hough_lines(sample_frame_path, length_threshold=100)

# Re-run the fixed aggregation process
aggregated_lines_fixed = aggregate_hough_lines_across_frames_fixed(output_frames_path)

# Visualize the aggregated lines on the first frame
visualize_aggregated_lines(os.path.join(output_frames_path, "frame_0076.png"), aggregated_lines_fixed)

# Estimate depth for the first extracted frame as a test
#sample_frame_path = os.path.join(output_frames_path, "frame_0000.png")
#depth_map_output = estimate_depth(sample_frame_path, depth_maps_path)
#depth_map_output

