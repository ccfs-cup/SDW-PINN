import gstools as gs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Define the bounds and size of the structured grid
x = y = np.linspace(0, 99, 51)

# Create a 2D Gaussian model with variance 1 and length scale 10
model = gs.Gaussian(dim=2, var=2, len_scale=28)

# Create the random field object
srf = gs.SRF(model)

# Generate a Gaussian random field
srf((x, y), mesh_type="structured")

# Transform the Gaussian random field into a lognormal random field
# Use np.exp to ensure positive values and obtain a lognormal random field
srf.field = np.exp(srf.field)
# Plot the 2D lognormal random field
plt.imshow(srf.field, cmap="jet", origin="lower", extent=[0, 51, 0, 51])
plt.colorbar()
plt.title("Original Log-Normal Permeability Field")
plt.savefig("PINN/random_field_generation/Original_field.png", dpi=300)  # Save the image
plt.close()
print("Conductivity field image before interpolation saved to 'Original_field.png'")

# Create a DataFrame using srf.field as the data
df = pd.DataFrame(srf.field)

# Save the DataFrame to an Excel file
excel_file = "PINN/random_field_generation/Original_field.xlsx"
df.to_excel(excel_file, index=False, header=False)
print(f"Conductivity field data before interpolation saved to {excel_file}")

# Create a new dense grid, e.g. 200x200
x_new = y_new = np.linspace(0, 99, 102)
# Create the interpolation function
interp_spline = RectBivariateSpline(x, y, srf.field)
# Interpolate on the new grid
field_interp = interp_spline(x_new, y_new)


# Plot the interpolated 2D lognormal random field
plt.imshow(field_interp, cmap="jet", origin="lower", extent=[0, 51, 0, 51])
plt.colorbar()
plt.title("Interpolated Log-Normal Permeability Field")
plt.savefig("PINN/random_field_generation/Interpolated_field.png", dpi=300)  # Save the image
plt.close()
print("Conductivity field image after interpolation saved to 'Interpolated_field.png'")

# Create a DataFrame using srf.field as the data
df = pd.DataFrame(field_interp)

# Save the DataFrame to an Excel file
excel_file = "PINN/random_field_generation/Interpolated_field.xlsx"
df.to_excel(excel_file, index=False, header=False)
print(f"Conductivity field data after interpolation saved to {excel_file}")
