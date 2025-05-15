import cv2, os, config
import numpy as np
import matplotlib.pyplot as plt

input_image_plasma = os.path.join(config.STEREO_OUTPUT_IMAGE, "balcony-stereo-plasma.jpg")
input_image_gray = os.path.join(config.STEREO_OUTPUT_IMAGE, "balcony-stereo-grey.jpg")
output_image_grid = os.path.join(config.STEREO_OUTPUT_IMAGE, "balcony-stereo-grid.jpg")

# Load images
disparity = cv2.imread(input_image_plasma, cv2.IMREAD_GRAYSCALE)
edges = cv2.imread(input_image_gray, cv2.IMREAD_GRAYSCALE)

# Resize for grid conversion (optional)
grid_size = 20  # example grid resolution
resized_disp = cv2.resize(disparity, (grid_size, grid_size))
resized_edges = cv2.resize(edges, (grid_size, grid_size))

grid = np.zeros((grid_size, grid_size), dtype=int)

# Thresholds to decide region types
floor_threshold = 15    # lower depth = closer (likely floor)
edge_threshold = 50     # edge strength

for y in range(grid_size):
    for x in range(grid_size):
        d = resized_disp[y, x]
        e = resized_edges[y, x]
        if d > floor_threshold:
            grid[y, x] = 1  # Floor
        if e > edge_threshold:
            grid[y, x] = 2  # Wall/Obstacle

plt.imshow(grid, cmap='tab20')
plt.title("2D Grid Layout (Floor/Wall Projection)")
plt.colorbar()
plt.savefig(output_image_grid)
plt.show()



