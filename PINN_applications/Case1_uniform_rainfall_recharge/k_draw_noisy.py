import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = '/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/K1.xlsx'  # Replace with the actual file path
xls = pd.ExcelFile(file_path)

# Load data from the first worksheet
data = pd.read_excel(xls, sheet_name='Sheet1', header=None)

# Convert the data to a NumPy array for plotting
data_values = data.values

# Add Gaussian noise
mean = 0.1  # Mean of the Gaussian noise
std_dev = 0.15  # Standard deviation of the Gaussian noise (adjust as needed)
noise = np.random.normal(mean, std_dev, data_values.shape)

# Add the noise to the data
noisy_data = data_values + noise

# Save the noisy data to a new Excel file
noisy_data_df = pd.DataFrame(noisy_data)

# Define the new Excel file path
output_file_path = '/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/K1_noisy.xlsx'

# Save to a new Excel file
noisy_data_df.to_excel(output_file_path, index=False, header=False)

# Create a heatmap
plt.figure(figsize=(8, 6.5))
heatmap = plt.imshow(noisy_data, cmap='jet', aspect='auto', extent=[0, 100, 0, 100])

# Add a colorbar and labels
cbar = plt.colorbar(heatmap)
cbar.ax.tick_params(labelsize=15)

# Set axis labels and ticks
plt.tick_params(axis='both', which='major', labelsize=15, width=1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Save and display the image
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/k_noisy.png', dpi=300)
