import json
import os
from src.core.image_processor import process_floorplan_image
from src.core.video_processor import process_video
from src.core.visualize_layout import visualize_layout
from config import INPUT_DIR, OUTPUT_DIR, FLOORPLAN_IMAGE
from src.core.numpy_encoder import NumpyEncoder  # Import the custom encoder


def image_main():
    image_file_path = FLOORPLAN_IMAGE
    output_image_path = os.path.join(OUTPUT_DIR, "visualization.png")
    parsed_image_path = os.path.join(OUTPUT_DIR, "parsed_image_layout.json")

    try:
        # Process the image
        parsed_image_data = process_floorplan_image(image_file_path, output_image_path)
        # Save the parsed layout to JSON using the custom encoder
        with open(parsed_image_path, "w") as f:
            json.dump(parsed_image_data, f, indent=4, cls=NumpyEncoder)
        print(f"Parsed image layout saved to {parsed_image_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def video_main():
    video_file_path = os.path.join(INPUT_DIR, "layout_video.mp4")  # Video file
    video_output_folder = os.path.join(OUTPUT_DIR, "video_frames")
    parsed_video_layout_path = os.path.join(OUTPUT_DIR, "parsed_video_layout.json")

    try:
        # Process the video
        parsed_video_data = process_video(video_file_path, video_output_folder)

        # Save the aggregated layout data to JSON
        with open(parsed_video_layout_path, "w") as f:
            json.dump(parsed_video_data, f, indent=4)
        print(f"Parsed video layout saved to {parsed_video_layout_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def visualize_video_layout():
    parsed_video_layout_path = os.path.join(OUTPUT_DIR, "parsed_video_layout.json")
    try:
        visualize_layout(parsed_video_layout_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    #image_main()
    #video_main()
    visualize_video_layout()
