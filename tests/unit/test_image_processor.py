from src.core.image_processor import process_floorplan_image

def test_process_floorplan_image():
    result = process_floorplan_image("data/input/img.png", "data/output/test_visualization.png")
    assert "walls" in result
    assert isinstance(result["walls"], list)
