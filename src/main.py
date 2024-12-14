import json
import os
from src.core.image_processor import process_floorplan_image
from config import INPUT_DIR, OUTPUT_DIR, FLOORPLAN_IMAGE
from src.core.numpy_encoder import NumpyEncoder  # Import the custom encoder

def main():
    # Define input and output paths
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

if __name__ == "__main__":
    main()
