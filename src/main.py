import json
from src.core.cad_parser import parse_cad_file
from src.core.image_processor import process_floorplan_image

def main():
    # CAD File Example
    cad_file_path = "data/input/floorplan.dxf"
    parsed_cad_data = parse_cad_file(cad_file_path)
    with open("data/output/parsed_cad_layout.json", "w") as f:
        json.dump(parsed_cad_data, f, indent=4)
    print("Parsed CAD layout saved to data/output/parsed_cad_layout.json")

    # Image File Example
    image_file_path = "data/input/floorplan.png"
    output_image_path = "data/output/visualization.png"
    parsed_image_data = process_floorplan_image(image_file_path, output_image_path)
    with open("data/output/parsed_image_layout.json", "w") as f:
        json.dump(parsed_image_data, f, indent=4)
    print("Parsed image layout saved to data/output/parsed_image_layout.json")

if __name__ == "__main__":
    main()
