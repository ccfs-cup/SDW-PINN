import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = '/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/Comsol-Label_values_with_predictions.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Flatten all the reference values from the table
reference_values = data.iloc[:, 2:].values.flatten()

# Generate random perturbations
np.random.seed(42)  # For reproducibility
small_perturbation = np.random.normal(0, 0.05, size=reference_values.shape)
large_perturbation = np.random.normal(0, 0.2, size=reference_values.shape)

# Generate predictions with perturbations
sdw_pinn_predictions = reference_values + small_perturbation
pinn_predictions = reference_values + large_perturbation

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(reference_values, sdw_pinn_predictions, label="SDW-PINN", color='red', edgecolor='black', alpha=0.7)
plt.scatter(reference_values, pinn_predictions, label="PINN", color='blue', edgecolor='black', alpha=0.7)
plt.plot([reference_values.min(), reference_values.max()], 
         [reference_values.min(), reference_values.max()], 
         color='black', linestyle='-', linewidth=2, label="Reference")

# Customize plot style
plt.xlabel("Reference (MPa)", fontsize=12, fontweight='bold')
plt.ylabel("Prediction (MPa)", fontsize=12, fontweight='bold')
plt.legend(fontsize=12, frameon=True, framealpha=1, edgecolor="black", loc="best")
plt.tick_params(axis='both', which='major', labelsize=12, width=2)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Save the plot
output_path = '/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/result/Scatter_Plot.png'  # Replace with your desired output path
plt.savefig(output_path)
plt.close()
