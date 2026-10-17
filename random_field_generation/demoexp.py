import gstools as gs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define the bounds and size of the structured grid
x = y = np.arange(100)

# Create a 2D Gaussian model with variance 1 and length scale 10
model = gs.Gaussian(dim=2, var=5, len_scale=30)

# Create the random field object
srf = gs.SRF(model)

# Generate a Gaussian random field
srf((x, y), mesh_type="structured")

# Transform the Gaussian random field into a lognormal random field
# Use np.exp to ensure positive values and obtain a lognormal random field
srf.field = np.exp(srf.field)


# Create a DataFrame using srf.field as the data
df = pd.DataFrame(srf.field)

# Save the DataFrame to an Excel file
excel_file = "/home/cc/CCFs/LYF/PINN/random_field_generation/Result/permeability_field_lognormal.xlsx"
df.to_excel(excel_file, index=False, header=False)
print(f"对数正态分布的渗流场数据已保存到 {excel_file}")

# Plot the 2D lognormal random field
plt.imshow(srf.field, cmap="jet", origin="lower")
plt.colorbar()
# plt.show()

# Save the image to a file
image_file = "/home/cc/CCFs/LYF/PINN/random_field_generation/Result/permeability_field_lognormal.png"  # Specify the filename for saving the image
plt.savefig(image_file, dpi=300)  # Save the image with a higher dpi for better quality
print(f"对数正态分布的渗流场图像已保存到 {image_file}")
# Close the figure to avoid displaying images on the server
plt.close()
