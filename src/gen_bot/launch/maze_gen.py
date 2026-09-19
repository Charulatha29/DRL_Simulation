# save as generate_3x3_grid.py
import cv2
import numpy as np

# Create white image (high resolution for better quality)
width, height = 3000, 3000  # 1000 pixels per meter
img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Grid parameters - 3x3 grid means 4 lines each direction
grid_divisions = 3
line_thickness = 15  # Black tape thickness (adjustable)
line_color = (0, 0, 0)  # Black

# Calculate spacing for 3 divisions (creating a 3x3 grid)
cell_size = width // grid_divisions

# Draw vertical lines (4 lines for 3 divisions)
for i in range(grid_divisions + 1):
    x = i * cell_size
    cv2.line(img, (x, 0), (x, height), line_color, line_thickness)

# Draw horizontal lines (4 lines for 3 divisions)
for i in range(grid_divisions + 1):
    y = i * cell_size
    cv2.line(img, (0, y), (width, y), line_color, line_thickness)

# Save the image
cv2.imwrite('grid_3x3_texture.png', img)
print("3x3 Grid texture created!")
print(f"Grid cell size: {cell_size} pixels (1 meter)")