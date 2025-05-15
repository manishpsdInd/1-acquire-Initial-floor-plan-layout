from src.core.phase1.cad_parser import parse_cad_file

def test_parse_cad_file():
    result = parse_cad_file("data/input/floorplan.dxf")
    assert "walls" in result
    assert isinstance(result["walls"], list)
