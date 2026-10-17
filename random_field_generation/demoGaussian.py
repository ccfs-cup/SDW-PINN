import gstools as gs
import pandas as pd
import numpy as np  # Import numpy to convert NumPy arrays to pandas DataFrames later
from matplotlib import pyplot as plt

# Define the bounds and size of the structured grid
x = y = np.arange(100)  # Use numpy.arange to create an arithmetic sequence

# Create a 2D Gaussian model with variance 1 and length scale 10
model = gs.Gaussian(dim=2, var=1, len_scale=10)

# Create the random field object
srf = gs.SRF(model)

# Generate a random field
srf((x, y), mesh_type="structured")

# Create a DataFrame using srf.field as the data
# Note: srf.field is a 2D array and can be converted directly to a DataFrame
df = pd.DataFrame(srf.field)

# Save the DataFrame to an Excel file
excel_file = "/home/cc/CCFs/LYF/PINN/random_field_generation/Result/permeability_field.xlsx"
df.to_excel(excel_file, index=False, header=False)

print(f"渗流场数据已保存到 {excel_file}")

# Slice srf to obtain a 1D array; for a 2D field, plotting part of the field may be more meaningful
# If a 1D slice is needed, verify that the slicing method meets your requirements
# Commented out here because this may not be the intended plot
# srf = srf[0, :, :]

# Plot the 2D random field
plt.imshow(srf.field, cmap="jet", origin="lower")  # Use imshow
plt.colorbar()  # Add a colorbar
# plt.show()
# Save the image to a file
image_file = "/home/cc/CCFs/LYF/PINN/random_field_generation/Result/permeability_field_lognormal.png"  # Specify the filename for saving the image
plt.savefig(image_file, dpi=300)  # Save the image with a higher dpi for better quality
print(f"对数正态分布的渗流场图像已保存到 {image_file}")
# Close the figure to avoid displaying images on the server
plt.close()
