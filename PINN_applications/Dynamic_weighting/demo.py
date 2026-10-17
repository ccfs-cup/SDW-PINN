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

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(permeability_field, cmap='viridis')
plt.colorbar(label='Permeability Value')
plt.title('Original Field')

plt.subplot(132)
plt.imshow(np.abs(grad_x), cmap='viridis', vmin=0, vmax=0.3)
plt.colorbar(label='Derivative Magnitude')
plt.title('Scharr Derivative')

plt.subplot(133)
plt.imshow(np.abs(diff_x), cmap='viridis', vmin=0, vmax=0.3)
plt.colorbar(label='Derivative Magnitude')
plt.title('Numerical Derivative')

plt.tight_layout()
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/scharr_vs_numerical.png', dpi=300)  # Save the image
# plt.show()