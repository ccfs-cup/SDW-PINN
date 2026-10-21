import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for custom legend entries

# Read the first Excel file (SDW-PINN data)
file_path_sdw = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Labeldata_with_predictions_SDW-PINN.xlsx'
data_sdw = pd.read_excel(file_path_sdw)

# Read the second Excel file (PINN data)
file_path_pinn = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Labeldata_with_predictions_PINN.xlsx'
data_pinn = pd.read_excel(file_path_pinn)

# Extract reference and predicted value columns
reference_columns = [col for col in data_sdw.columns if col.startswith('t=')]
prediction_columns_sdw = [col for col in data_sdw.columns if col.startswith('Prediction_t=')]
prediction_columns_pinn = [col for col in data_pinn.columns if col.startswith('Prediction_t=')]

all_references = []
all_predictions_sdw = []
all_predictions_pinn = []

# Aggregate the data
for ref_col, pred_col_sdw, pred_col_pinn in zip(reference_columns, prediction_columns_sdw, prediction_columns_pinn):
    all_references.extend(data_sdw[ref_col].values)
    all_predictions_sdw.extend(data_sdw[pred_col_sdw].values)
    all_predictions_pinn.extend(data_pinn[pred_col_pinn].values)

all_references = np.array(all_references)
all_predictions_sdw = np.array(all_predictions_sdw)
all_predictions_pinn = np.array(all_predictions_pinn)

# Randomly sample 200 points
sample_size = 200
total_points = len(all_references)
sample_indices = np.random.choice(total_points, sample_size, replace=False)

sample_references = all_references[sample_indices]
sample_predictions_sdw = all_predictions_sdw[sample_indices]
sample_predictions_pinn = all_predictions_pinn[sample_indices]

# Plot the correlation between reference and predicted values
plt.figure(figsize=(8, 6))

# Plot PINN data as hollow blue triangles
plt.scatter(
    sample_references, 
    sample_predictions_pinn, 
    edgecolors='blue',   # Edge color
    facecolors='none',   # No fill
    alpha=0.8, 
    label='PINN', 
    marker='^',          # Triangle markers
    s=100,               # Set the scatter marker size
    linewidths=1.5       # Set the edge width
)

# Plot SDW-PINN data as hollow red circles
plt.scatter(
    sample_references, 
    sample_predictions_sdw, 
    edgecolors='red',   # Edge color
    facecolors='none',  # No fill
    alpha=0.8, 
    label='SDW-PINN', 
    marker='o',
    s=100,              # Set the scatter marker size
    linewidths=1.5      # Set the edge width
)



# Plot the diagonal line
min_val = 48.1
max_val = 110
plt.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=2.5, label='_nolegend_')

# Set axis limits
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.gca().set_aspect('equal', adjustable='box')

# Set axis labels and legend
plt.xlabel(r'Reference', fontsize=15)
plt.ylabel(r'Prediction', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15, width=1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)

# Use Line2D to add a custom legend entry for the 45-degree line
legend_elements = [Line2D([0], [0], color='k', lw=2.5, label='Simulator')]
# Combine and display the legend entries
plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], fontsize=12)
# plt.legend(fontsize=12)

# Adjust the layout and save the image
plt.tight_layout()
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/CorrelationPlot_MultiModel.png', dpi=300)

# Display the image
plt.show()
