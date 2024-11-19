import sys
import ezdxf

try:
    doc = ezdxf.readfile("data/input/floorplan.png")
except IOError:
    print(f"Not a DXF file or a generic I/O error.")
    sys.exit(1)
except ezdxf.DXFStructureError:
    print(f"Invalid or corrupted DXF file.")
    sys.exit(2)