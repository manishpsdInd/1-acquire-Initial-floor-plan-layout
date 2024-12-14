import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):  # Handle NumPy integers
            return int(obj)
        if isinstance(obj, (np.floating, float)):  # Handle NumPy floats
            return float(obj)
        if isinstance(obj, np.ndarray):  # Handle NumPy arrays
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
