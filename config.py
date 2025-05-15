import os

# Define the root path of the project dynamically
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Define paths for input and output directories relative to the root path
INPUT_DIR = os.path.join(ROOT_PATH, "data/input")
OUTPUT_DIR = os.path.join(ROOT_PATH, "data/output")

# Example specific file paths
FLOORPLAN_IMAGE = os.path.join(INPUT_DIR, "floorplan.png")
FLOORPLAN_DXF = os.path.join(INPUT_DIR, "floorplan.dxf")

# Paths for video processing
VIDEO_FILE = os.path.join(INPUT_DIR, "layout_video.mp4")

STEREO_INPUT_IMAGE  = os.path.join(ROOT_PATH, "data/input/sample-images")
STEREO_OUTPUT_IMAGE = os.path.join(ROOT_PATH, "data/output/generated-images")
