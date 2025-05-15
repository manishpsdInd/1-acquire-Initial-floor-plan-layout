import json
import matplotlib.pyplot as plt

def visualize_layout(json_path, output_path=None):
    """
    Visualizes the room layout based on the JSON file containing wall coordinates.
    :param json_path: Path to the JSON file with parsed layout data
    :param output_path: Path to save the visualization image (optional)
    """
    # Load the JSON file
    with open(json_path, "r") as f:
        layout_data = json.load(f)

    # Create a new plot
    plt.figure(figsize=(8, 8))
    plt.title("Room Layout Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot the walls
    walls = layout_data.get("walls", [])
    for wall in walls:
        start = wall["start"]
        end = wall["end"]
        plt.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=2)

    # Set the aspect ratio to be equal
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Path to the generated JSON file
    json_file = "data/output/parsed_video_layout.json"
    # Optional: Path to save the visualization
    visualization_output = "data/output/room_layout_visualization.png"

    # Visualize the layout
    visualize_layout(json_file, visualization_output)
