import ezdxf

def parse_cad_file(file_path):
    """
    Parses a CAD file (DXF format) to extract wall, door, and furniture elements.
    :param file_path: Path to the DXF file
    :return: Parsed layout as a dictionary
    """
    layout_data = {"walls": [], "doors": [], "furniture": []}
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    # Extract lines (assumed to represent walls)
    for line in msp.query("LINE"):
        start = (line.dxf.start.x, line.dxf.start.y)
        end = (line.dxf.end.x, line.dxf.end.y)
        layout_data["walls"].append({"start": start, "end": end})

    # Extract other elements like doors or furniture if available
    for block in msp.query("INSERT"):
        if block.dxf.name.lower() == "door":
            layout_data["doors"].append({"position": (block.dxf.insert.x, block.dxf.insert.y)})
        elif block.dxf.name.lower() == "furniture":
            layout_data["furniture"].append({"position": (block.dxf.insert.x, block.dxf.insert.y)})

    return layout_data
