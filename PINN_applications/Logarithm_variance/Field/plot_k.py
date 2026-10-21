import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Excel file
file_path = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.5.xlsx'  # Replace with the actual path to your Excel file
xls = pd.ExcelFile(file_path)

# Load the data from the first sheet
data = pd.read_excel(xls, sheet_name='Sheet1')

# Convert the data to a numpy array for plotting
data_values = data.values

# Create a mesh grid for the range 0 to 100
x = np.arange(101)
y = np.arange(101)
X, Y = np.meshgrid(x, y)

# Create a heatmap with the given data
plt.figure(figsize=(8, 6.5))
heatmap = plt.imshow(data_values, cmap='jet', aspect='auto', extent=[0, 100, 0, 100], origin='lower', vmin=2.1, vmax=38)
#   plt.imshow(srf.field, cmap="jet", origin="lower", vmin=2.1, vmax=38)  # Set the display range

# Adding color bar and labels
# plt.colorbar(label='Value')
# Adding color bar and labels
cbar = plt.colorbar(heatmap)
cbar.set_label(r'$H/m$', fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

plt.tick_params(axis='both', which='major', labelsize=20, width=1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
# plt.title('Heatmap of the Data')
plt.xlabel(r'$x/m$', fontsize=20, fontweight='bold')
plt.ylabel(r'$y/m$', fontsize=20, fontweight='bold')

# Save and show the plot
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.5.png')

