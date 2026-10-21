import gstools as gs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define the bounds and size of the structured grid
x = y = np.arange(101)

# Create a 2D Gaussian model with variance 5 and length scale 30
model = gs.Gaussian(dim=2, var=8, len_scale=10)

# Define the target range of random field values
min_value = 1
max_value = 100

for i in range(5):  # Generate 20 different random fields
    # Create the random field object
    srf = gs.SRF(model)

    # Generate a Gaussian random field
    # srf((x, y), mesh_type="structured", seed=i)  # Use different seeds to ensure randomness
    srf((x, y), mesh_type="structured")  # Use different seeds to ensure randomness

    # Transform the Gaussian random field into a lognormal random field
    # Use np.exp to ensure positive values and obtain a lognormal random field
    srf.field = np.exp(srf.field)

    # Limit the random field values to the specified range
    srf.field = (srf.field - srf.field.min()) / (srf.field.max() - srf.field.min())  # Normalize to [0, 1]
    srf.field = srf.field * (max_value - min_value) + min_value  # Scale to [min_value, max_value]

    # Create a DataFrame using srf.field as the data
    df = pd.DataFrame(srf.field)

    # Save the DataFrame to an Excel file
    excel_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/xslx/permeability_field_lognormal_{i}.xlsx"
    df.to_excel(excel_file, index=False, header=False)
    print(f"Lognormal conductivity field data saved to {excel_file}")

    # Plot the 2D lognormal random field
    plt.imshow(srf.field, cmap="jet", origin="lower", vmin=min_value, vmax=max_value)  # Set the display range
    plt.colorbar()

    # Save the image to a file
    image_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/permeability_field_lognormal_{i}.png"
    plt.savefig(image_file, dpi=300)
    print(f"Lognormal conductivity field image saved to {image_file}")

    # Close the figure to avoid displaying images on the server
    plt.close()
