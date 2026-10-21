import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Excel file
file_path = '/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/Comsol-Label_values_with_predictions.xlsx'
data = pd.read_excel(file_path)

# Extract all reference and predicted values
reference_columns = [col for col in data.columns if col.startswith('t=')]
prediction_columns = [col for col in data.columns if col.startswith('Prediction_t=')]

all_references = []
all_predictions = []

for ref_col, pred_col in zip(reference_columns, prediction_columns):
    all_references.extend(data[ref_col].values)
    all_predictions.extend(data[pred_col].values)

all_references = np.array(all_references)
all_predictions = np.array(all_predictions)

# Randomly sample 200 points
sample_size = 200
total_points = len(all_references)
sample_indices = np.random.choice(total_points, sample_size, replace=False)

sample_references = all_references[sample_indices]
sample_predictions = all_predictions[sample_indices]

# Plot the correlation between reference and predicted values
plt.figure(figsize=(6, 6))
plt.scatter(
    sample_references, 
    sample_predictions, 
    edgecolors='red',   # Edge color
    facecolors='none',  # No fill
    alpha=0.8, 
    label='SDW-PINN', 
    marker='o',
    s=100,              # Set the scatter marker size
    linewidths=1.5      # Set the edge width
)
min_val = 75
max_val = 105
# plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5, label='y = x')
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5, label='_nolegend_')

# Set axis limits without extra whitespace
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# Set the legend and axis labels
# plt.xlabel('Reference', fontsize=15)
# plt.ylabel('Prediction', fontsize=15)
plt.xlabel(r'Reference', fontsize=15)
plt.ylabel(r'Prediction', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15, width=1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.legend(fontsize=12)

# Adjust the layout and save the image
plt.tight_layout()
# plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/CorrelationPlot.png', dpi=300)

# Display the image
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/CorrelationPlot.png', dpi=300)
