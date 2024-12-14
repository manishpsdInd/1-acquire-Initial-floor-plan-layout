import ezdxf

def parse_cad_file(file_path):
    """
    Parses a CAD file (DXF format) to extract wall, door, and furniture elements.
    :param file_path: Path to the DXF file
    :return: Parsed layout as a dictionary
    """
    layout_data = {"walls": [], "doors": [], "furniture": []}

    # Load the DXF file
    try:
        doc = ezdxf.readfile(file_path)
    except ezdxf.DXFError as e:
        raise ValueError(f"Error reading DXF file: {file_path}, {str(e)}")

    msp = doc.modelspace()

    # Extract lines (assumed to represent walls)
    for line in msp.query("LINE"):
        start = (line.dxf.start.x, line.dxf.start.y)
        end = (line.dxf.end.x, line.dxf.end.y)
        layout_data["walls"].append({"start": start, "end": end})

    # Extract block inserts (e.g., doors, furniture)
    for insert in msp.query("INSERT"):
        name = insert.dxf.name.lower()
        position = (insert.dxf.insert.x, insert.dxf.insert.y)
        if "door" in name:
            layout_data["doors"].append({"position": position})
        elif "furniture" in name:
            layout_data["furniture"].append({"position": position})

    return layout_data
