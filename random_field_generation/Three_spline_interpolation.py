import gstools as gs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Define the bounds and size of the structured grid
x = y = np.linspace(0, 99, 100)

# Create a 2D Gaussian model with variance 1 and length scale 10
model = gs.Gaussian(dim=2, var=1, len_scale=28)

# Create the random field object
srf = gs.SRF(model)

# Generate a Gaussian random field
srf((x, y), mesh_type="structured")

# Transform the Gaussian random field into a lognormal random field
# Use np.exp to ensure positive values and obtain a lognormal random field
srf.field = np.exp(srf.field)
# Plot the 2D lognormal random field
plt.imshow(srf.field, cmap="jet", origin="lower")
plt.colorbar()
plt.title("Original Log-Normal Permeability Field")
plt.savefig("Original_field.png", dpi=300)  # Save the image
plt.close()
print("插值前的渗透系数图像已保存到 'Original_field.png'")

# Create a DataFrame using srf.field as the data
df = pd.DataFrame(srf.field)

# Create a new dense grid, e.g. 200x200
x_new = y_new = np.linspace(0, 99, 200)
# Create the interpolation function
interp_spline = RectBivariateSpline(x, y, srf.field)
# Interpolate on the new grid
field_interp = interp_spline(x_new, y_new)

# Plot the interpolated 2D lognormal random field
plt.imshow(field_interp, cmap="jet", origin="lower")
plt.colorbar()
plt.title("Interpolated Log-Normal Permeability Field")
plt.savefig("Interpolated_field.png", dpi=300)  # Save the image
plt.close()
print("插值后的渗透系数图像已保存到 'Interpolated_field.png'")
