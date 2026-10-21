import pandas as pd
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Read the Excel data
df = pd.read_excel("/home/users/liyifan/LYF/PINN_applications/Correlation_length/Field/K30.xlsx", header=None)
permeability_field = df.values

# Scharr operator (scale factor 1/32)
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]) / 32
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]) / 32

# Calculate derivatives
grad_x = convolve2d(permeability_field, scharr_x, mode='same')
grad_y = convolve2d(permeability_field, scharr_y, mode='same')

# Numerical differences
diff_x = np.zeros_like(permeability_field)
diff_x[:, 1:-1] = (permeability_field[:, 2:] - permeability_field[:, :-2]) / 2

# Calculate the error (absolute difference)
error = np.abs(grad_x - diff_x)

# Visualize in a 2-row, 2-column layout
plt.figure(figsize=(15, 8))  # Increase the figure size to accommodate 4 subplots

plt.subplot(221)
plt.imshow(permeability_field, cmap='viridis')
plt.colorbar(label='Permeability Value')
plt.title('Original Field')

plt.subplot(222)
plt.imshow(np.abs(grad_x), cmap='viridis', vmin=0, vmax=0.5)
plt.colorbar(label='Derivative Magnitude')
plt.title('Scharr Derivative')

plt.subplot(223)
plt.imshow(np.abs(diff_x), cmap='viridis', vmin=0, vmax=0.5)
plt.colorbar(label='Derivative Magnitude')
plt.title('Numerical Derivative')

plt.subplot(224)
plt.imshow(error, cmap='viridis', vmin=0, vmax=0.3)  # Use the same color range as the derivative plots
plt.colorbar(label='Error Magnitude')
plt.title('Error (Scharr - Numerical)')

plt.tight_layout()
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/scharr_vs_numerical.png', dpi=300)
# plt.show()