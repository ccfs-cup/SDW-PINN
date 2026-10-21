# import gstools as gs
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt

# # Define the bounds and size of the structured grid
# x = y = np.arange(101)

# # Define the lognormal distribution parameters
# log_mean = 10
# log_variance = 0.1
# correlation_length = 30

# # Convert the log-space parameters to actual parameters
# mean = np.log(log_mean)
# variance = log_variance
# std_dev = np.sqrt(variance)

# # Create a 2D Gaussian model using the variance and correlation length
# model = gs.Gaussian(dim=2, var=variance, len_scale=correlation_length)

# # Define the target range of random field values
# min_value = 0
# max_value = 20

# for i in range(15):  # Generate 5 different random fields
#     # Create the random field object
#     srf = gs.SRF(model)

#     # Generate a Gaussian random field
#     srf((x, y), mesh_type="structured")  # Use different seeds to ensure randomness

#     # Transform the Gaussian random field into a lognormal random field
#     # Add the log-space mean and use np.exp to ensure positive values
#     srf.field = np.exp(srf.field + mean)

#     # Limit the random field values to the specified range
#     srf.field = (srf.field - srf.field.min()) / (srf.field.max() - srf.field.min())  # Normalize to [0, 1]
#     srf.field = srf.field * (max_value - min_value) + min_value  # Scale to [min_value, max_value]

#     # Create a DataFrame using srf.field as the data
#     df = pd.DataFrame(srf.field)

#     # Save the DataFrame to an Excel file
#     excel_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/xslx/permeability_field_lognormal_{i}.xlsx"
#     df.to_excel(excel_file, index=False, header=False)
#     print(f"Lognormal permeability field data saved to {excel_file}")

#     # Plot the 2D lognormal random field
#     plt.imshow(srf.field, cmap="jet", origin="lower", vmin=min_value, vmax=max_value)  # Set the display range
#     plt.colorbar()

#     # Save the image to a file
#     image_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/permeability_field_lognormal_{i}.png"
#     plt.savefig(image_file, dpi=300)
#     print(f"Lognormal permeability field image saved to {image_file}")

#     # Close the figure to avoid displaying images on the server
#     plt.close()
import gstools as gs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define the bounds and size of the structured grid
x = y = np.arange(101)

# Define the lognormal distribution parameters
log_mean = 10
log_variance = 0.09
correlation_length = 30

# Convert the log-space parameters to actual parameters
mean = np.log(log_mean)
variance = log_variance
std_dev = np.sqrt(variance)

# Create a 2D Gaussian model using the variance and correlation length
model = gs.Gaussian(dim=2, var=variance, len_scale=correlation_length)

# Generate a random field to calculate the dynamic range
srf_sample = gs.SRF(model)
srf_sample((x, y), mesh_type="structured")
srf_sample.field = np.exp(srf_sample.field + mean)  # Transform to a lognormal distribution

# Calculate min_value and max_value dynamically
min_value = np.min(srf_sample.field)
max_value = np.max(srf_sample.field)


print(f"Dynamically calculated target range: min_value = {min_value}, max_value = {max_value}")

for i in range(15):  # Generate 15 different random fields
    # Create the random field object
    srf = gs.SRF(model, seed=i)  # Use different seeds to ensure randomness

    # Generate a Gaussian random field
    srf((x, y), mesh_type="structured")

    # Transform the Gaussian random field into a lognormal random field
    srf.field = np.exp(srf.field + mean)

    # Limit the random field values to the dynamically calculated range
    srf.field = (srf.field - srf.field.min()) / (srf.field.max() - srf.field.min())  # Normalize to [0, 1]
    srf.field = srf.field * (max_value - min_value) + min_value  # Scale to the dynamic range

    # Create a DataFrame using srf.field as the data
    df = pd.DataFrame(srf.field)

    # Save the DataFrame to an Excel file
    excel_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/xslx/permeability_field_lognormal_{i}.xlsx"
    df.to_excel(excel_file, index=False, header=False)
    print(f"Lognormal conductivity field data saved to {excel_file}")

    # Plot the 2D lognormal random field
    plt.imshow(srf.field, cmap="jet", origin="lower", vmin=min_value, vmax=max_value)  # Set the display range
    # plt.imshow(srf.field, cmap="jet", origin="lower", vmin=2.1, vmax=38)  # Set the display range
    plt.colorbar()

    # Save the image to a file
    image_file = f"/home/users/liyifan/LYF/PINN/random_field_generation/Result/png/permeability_field_lognormal_{i}.png"
    plt.savefig(image_file, dpi=300)
    print(f"Lognormal conductivity field image saved to {image_file}")

    # Close the figure to avoid displaying images on the server
    plt.close()
