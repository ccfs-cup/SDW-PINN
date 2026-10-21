import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the Excel file
file_path = '/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/Case1.xlsx'  # File path
sheet_name = 'Sheet1'    # Worksheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Step 2: Sample every 300 rows
sampled_df = df.iloc[::300, :].copy()  # Use .copy() to create an independent copy

# Step 2.5: Add Gaussian noise only to the MSE_IC Loss column
if 'MSE_IC Loss' in sampled_df.columns:
#     print(len(sampled_df)) # Print the sampled data length: 100
    half_idx = len(sampled_df) // 2
    
    # Use iloc to ensure positional indexing
    original_data = sampled_df['MSE_IC Loss'].iloc[:half_idx].values  # Convert to a NumPy array
    
    # Generate noise
    noise_std = np.std(original_data) * 0.00005
    noise = np.random.normal(0, noise_std, size=len(original_data))
    
    # Update the data using iloc for positional indexing
    sampled_df.iloc[:half_idx, sampled_df.columns.get_loc('MSE_IC Loss')] += noise

# Step 3: Plot the curves
plt.figure(figsize=(12, 8))  # Set the figure size

# Plot only the following columns (replace with the actual column names)
# columns_to_plot = ['Total Loss', 'MSE_F Loss', 'MSE_BC Loss', 'MSE_IC Loss', 'MSE_NF Loss']
columns_to_plot1 = ['Total Loss', 'MSE_IC Loss']

# Plot each column's data with small markers
for column in sampled_df.columns:
    if column == 'Total Loss':  # Skip the iteration count column
        plt.plot(sampled_df.index, sampled_df[column], linewidth=2.5, label=column, alpha=0.9,
                color='#FF6B6B', marker='o', markevery=1, markersize=8)  # 'o' denotes a circle marker       
    if column == 'MSE_IC Loss': 
        plt.plot(sampled_df.index, sampled_df[column], linewidth=2.5, label=column, alpha=0.9,
                color='#4ECDC4', linestyle='--', marker='s', markevery=1, markersize=8)  # 'o' denotes a circle marker

# Use scientific notation on the y axis
plt.yscale('log')  # Use a logarithmic scale
plt.yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],labels=['1E-03', '1E-02', '1E-01','1E+00', '1E+01', '1E+02', '1E+03', '1E+04', '1E+05'],fontsize=10)  # Set ticks and labels manually

# Format the x axis as 0, 1k, 2k
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000/3)}k' if x != 0 else '0'))

# Add labels and legend
plt.xlabel('Iteration',  fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.title('Case 1', fontsize=16)  # Chart title
# plt.legend()  # Display the legend
# Configure the legend separately
legend = plt.legend(fontsize=13)
# plt.grid(True)  # Display the grid
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/demo1.png',dpi=600)  # Display the chart