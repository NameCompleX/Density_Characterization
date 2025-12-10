import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


characterize_PATH = "Your/Path/To/characterize.txt"
REAL_DIR = "Your/Path/To/Real/"
image_size = 512  
WIDTH = 4
LENGTH = 4


def read_data(file_path):
    # Load data from text: first 4 columns = coordinates, last column = density value
    data = np.loadtxt(file_path)
    return data[:, :4], data[:, 4]  

# Read predicted and real density data
characterize_coords, characterize_density = read_data(characterize_PATH)
real_coords, real_density = read_data(REAL_DIR)

# Initialize empty density maps
characterize_density_map = np.zeros((image_size, image_size))
real_density_map = np.zeros((image_size, image_size))

# Fill predicted density map using rectangular regions
for (x_start, x_end, y_start, y_end), density in zip(characterize_coords, characterize_density):
    characterize_density_map[int(y_start):int(y_end), int(x_start):int(x_end)] = density  

# Fill real density map using rectangular regions
for (x_start, x_end, y_start, y_end), density in zip(real_coords, real_density):
    real_density_map[int(y_start):int(y_end), int(x_start):int(x_end)] = density  

# Simple smoothing kernel (identity here, but kept for consistency)
kernel = np.ones((1, 1)) / 1  

# Apply convolution smoothing
characterize_density_map = convolve(characterize_density_map, kernel, mode='reflect')
real_density_map = convolve(real_density_map, kernel, mode='reflect')

# Compute pixel-wise density difference image
density_diff_image = characterize_density_map - real_density_map

# Recompute average predicted density values (not modifying map, but included)
for pred_coord, pred_value in zip(characterize_coords, characterize_density):
    x_start, x_end, y_start, y_end = map(int, pred_coord)

    # Ensure the region stays inside the image
    x_start, x_end = max(0, x_start), min(image_size, x_end)
    y_start, y_end = max(0, y_start), min(image_size, y_end)

    area = (x_end - x_start) * (y_end - y_start)
    avg_value = pred_value if area == 0 else pred_value / area  

    # Access map region (no operation performed)
    characterize_density_map[y_start:y_end, x_start:x_end]

# Same procedure for real density data
for real_coord, real_value in zip(real_coords, real_density):
    x_start, x_end, y_start, y_end = map(int, real_coord)
    x_start, x_end = max(0, x_start), min(image_size, x_end)
    y_start, y_end = max(0, y_start), min(image_size, y_end)

    area = (x_end - x_start) * (y_end - y_start)
    avg_value = real_value if area == 0 else real_value / area  

    real_density_map[y_start:y_end, x_start:x_end]


# --- Plot predicted, real, and diff maps ---

fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# Remove axes for clean visualization
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')

# Predicted density heatmap
im1 = axs[0].imshow(characterize_density_map, cmap="hot", interpolation="nearest")
fig.colorbar(im1, ax=axs[0],ticks=[0, 20, 40, 60, 80], fraction=0.05, pad=0.02)

# Real density heatmap
im2 = axs[1].imshow(real_density_map, cmap="hot", interpolation="nearest")
fig.colorbar(im2, ax=axs[1],ticks=[0, 20, 40, 60, 80], fraction=0.05, pad=0.02)

# Difference heatmap (error map)
im3 = axs[2].imshow(density_diff_image, cmap="hot", interpolation="nearest")
fig.colorbar(im3, ax=axs[2],ticks=[0, 20, 40, 60, 80], fraction=0.05, pad=0.02)

# Normalize color scaling for all heatmaps
for im in [im1, im2, im3]:
    im.set_clim(-1, 81)

plt.tight_layout()
plt.show()

# --- Plot histogram of pixel-wise density error ---

plt.figure(figsize=(4, 4))
plt.hist(density_diff_image.flatten(), bins=100, color='orange', alpha=0.7)

plt.title('Error Distribution', fontsize=10)
plt.xlabel('Density Difference Value', fontsize=10)
plt.xlim(-5, 5)
plt.gca().yaxis.set_ticks([])
plt.ylabel('Count (a.u.)', fontsize=10)
plt.tight_layout()
plt.show()
