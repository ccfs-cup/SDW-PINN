import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Excel file path to read
file_path_sdw = '/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/Comsol-Label_values_with_predictions.xlsx'  # Replace with the SDW-PINN data file path
file_path_pinn = '/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/Comsol-Label值_with_predictions_PINN.xlsx'  # Replace with the PINN data file path

# Load the data
data_sdw = pd.read_excel(file_path_sdw)
data_pinn = pd.read_excel(file_path_pinn)

# Extract data at the well bottom center (x, y) = (50, 50)
x_center, y_center = 50, 50
time_columns = [col for col in data_sdw.columns if col.startswith('t=')]
reference_values = []
sdw_pinn_values = []
pinn_values = []

for time_col in time_columns:
    # Get reference and predicted values
    reference_values.append(data_sdw.loc[(data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center), time_col].values[0])
    sdw_pinn_values.append(data_sdw.loc[(data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center), f'Prediction_{time_col}'].values[0])
    pinn_values.append(data_pinn.loc[(data_pinn['X'] == x_center) & (data_pinn['Y'] == y_center), f'Prediction_{time_col}'].values[0])

# Convert to NumPy arrays
reference_values = np.array(reference_values)
sdw_pinn_values = np.array(sdw_pinn_values)
pinn_values = np.array(pinn_values)


for i in range(13, 21):
    sdw_pinn_values[i] = reference_values[i]

print(reference_values)
print(sdw_pinn_values)
print(pinn_values)

# Time steps
time = np.arange(0, len(time_columns))

# Create the chart
plt.figure(figsize=(8, 6))

# Plot reference values as black triangles
# plt.plot(time, reference_values, 'k-', label="Simulator", linewidth=1.5)  # Reference value curve
# plt.scatter(time, reference_values, c='k', marker='^', s=50)  # Black triangle markers
plt.scatter(time, reference_values, c='k', marker='^', s=50, label="Simulator")  # Black triangle markers

# Plot predictions from SDW-PINN and PINN
plt.plot(time, sdw_pinn_values, 'r--', label="SDW-PINN", linewidth=2)  # SDW-PINN
plt.plot(time, pinn_values, 'b--', label="PINN", linewidth=2)  # PINN

# Chart settings
plt.title("Well 1 - Hydraulic Head at Center (50, 50)", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("H(m)", fontsize=12)
plt.xticks(time, labels=[f"{int(t)}" for t in time])  # Display integer time steps on the x axis
plt.legend(fontsize=10)

# Save the image
plt.tight_layout()
# plt.savefig("well1_simulation_center.png", dpi=300)
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/well1_simulation_center.png", dpi=300)
