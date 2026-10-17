import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Load the data
file_path = '/home/cc/CCFs/LYF/PINN/Source and sink term groundwater flow problem/Case7:承压含水层多抽水井(非均质各向异性)/井底水头值.xlsx'
data = pd.read_excel(file_path)

# # Convert both X and Y columns to integers (assuming the comparison points have integer coordinates)
# data['X'] = data['X'].astype(int)
# data['Y'] = data['Y'].astype(int)
data['X'] = data['X'].round().astype(int)
data['Y'] = data['Y'].round().astype(int)


# Define the points to plot and their well labels
points = [(30, 30), (50, 80), (70, 50)]
well_labels = ['Well 1', 'Well 2', 'Well 3']
time_steps = list(range(11))  # Time steps from t=0 to t=10

# Define 6 different markers and colors for the reference and predicted values at each well
markers = ['P', 'X', 'D', 'o', 'v', 's']  # 6 different markers
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # 6 different colors

# Initialize the figure
plt.figure(figsize=(10, 6))

# Plot reference and predicted values for each point
for i, (x, y) in enumerate(points):
    print(i, (x, y))
    # Filter the data at the corresponding location
    point_data = data[(data['X'] == x) & (data['Y'] == y)]
    
    if point_data.empty:
        print(f"No data for point ({x}, {y}), skipping.")
        continue  # Skip this point if no data is available
    
    # Extract reference and predicted values
    reference_values = point_data[[f't={i}' for i in time_steps]].values.flatten()
    prediction_values = point_data[[f'Prediction_t={i}' for i in time_steps]].values.flatten()
    print(reference_values)
    print(prediction_values)
    
    # Check whether data dimensions match
    if len(reference_values) != len(time_steps) or len(prediction_values) != len(time_steps):
        print(f"Data dimension mismatch for point ({x}, {y}), skipping.")
        continue  # Skip this point if the dimensions do not match
    
    # Plot reference curves using different markers and colors
    plt.plot(time_steps, reference_values, 
             label=f'{well_labels[i]} Reference Values ({x}, {y})', 
             marker=markers[i * 2], markersize=8, linewidth=2, color=colors[i * 2])
    
    # Plot predicted curves as dashed lines using different markers and colors
    plt.plot(time_steps, prediction_values, 
             label=f'{well_labels[i]} Prediction Values ({x}, {y})',  
             marker=markers[i * 2 + 1], markersize=8, linewidth=2, color=colors[i * 2 + 1], linestyle='--')

# Create a custom legend
custom_lines = [
    Line2D([0], [0], color='blue', marker='P', markersize=8, linewidth=2, label='Well 1 COMSOL Reference'),
    Line2D([0], [0], color='orange', marker='X', markersize=8, linewidth=2, linestyle='--', label='Well 1 SDW-PINN Prediction'),
    Line2D([0], [0], color='green', marker='D', markersize=8, linewidth=2, label='Well 2 COMSOL Reference'),
    Line2D([0], [0], color='red', marker='o', markersize=8, linewidth=2, linestyle='--', label='Well 2 SDW-PINN Prediction'),
    Line2D([0], [0], color='purple', marker='v', markersize=8, linewidth=2, label='Well 3 COMSOL Reference'),
    Line2D([0], [0], color='brown', marker='s', markersize=8, linewidth=2, linestyle='--', label='Well 3 SDW-PINN Prediction')
]

# Add a title, labels, and legend
plt.title('Reference vs Prediction Values for Multiple Wells', fontsize=18, fontweight='bold')
plt.xlabel('Time Step', fontsize=15, fontweight='bold')
plt.ylabel('Water Head Value', fontsize=15, fontweight='bold')
plt.legend(handles=custom_lines, fontsize=12)

# Set axis ticks and borders
plt.tick_params(axis='both', which='major', labelsize=15, width=2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# Display the grid and save the image
# plt.grid(True)
plt.savefig('/home/cc/CCFs/LYF/PINN/Source and sink term groundwater flow problem/Case7:承压含水层多抽水井(非均质各向异性)/Result/multiple-wells.png')
plt.close()



