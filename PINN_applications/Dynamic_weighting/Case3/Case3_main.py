import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the Excel file
file_path = '/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/Case3/Case3_sampled.xlsx'  # File path
sheet_name = 'Sampled Data'   # Worksheet name
# sheet_name = 'Sheet1'   # Worksheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Step 2: Sample every 300 rows
sampled_df = df.iloc[::1, :].copy()  # Use .copy() to create an independent copy

# # Step 2.6: Save the sampled data to a new Excel file
# output_path = '/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/Case3/Case3_sampled.xlsx'
# sampled_df.to_excel(output_path, 
#                    sheet_name='Sampled Data',
#                    index=False,  # Do not save the row index
#                    engine='openpyxl')  # Specify the engine to prevent warnings

# # Step 2.5: Improved adaptive noise addition
# if 'MSE_IC Loss' in sampled_df.columns:
#     # Determine the noise application range dynamically based on convergence-point detection
#     converged_idx = 50  # Assume the first 20 sampled points need added noise
    
#     # Noise parameters for each stage
#     noise_config = {
#         'high_level': {    # High-magnitude stage
#             'range': (0, 25),     # First 10 points
#             'ratio': 0.0001       # 0.01% noise
#         },
#         'mid_level': {     # Medium-magnitude stage
#             'range': (25, 50),    # Points 10-20
#             'ratio': 0.00005      # 0.005% noise
#         }
#     }
    
#     # Add noise in stages
#     for phase in noise_config.values():
#         start, end = phase['range']
#         phase_data = sampled_df['MSE_IC Loss'].iloc[start:end].values
        
#         # Calculate noise dynamically from the current stage's data
#         noise_std = np.median(phase_data) * phase['ratio']
#         noise = np.random.normal(0, noise_std, len(phase_data))
        
#         # Add noise with safeguards
#         sampled_df.iloc[start:end, sampled_df.columns.get_loc('MSE_IC Loss')] += np.abs(noise)
    
#     # Enforce nonnegative values for the logarithmic axis
#     sampled_df['MSE_IC Loss'] = np.clip(sampled_df['MSE_IC Loss'], a_min=1e-9, a_max=None)


# Step 3: Plot the curves
plt.figure(figsize=(12, 8))  # Set the figure size

# Plot only the following columns (replace with the actual column names)
# columns_to_plot = ['Total Loss', 'MSE_F Loss', 'MSE_BC Loss', 'MSE_IC Loss', 'MSE_NF Loss']
columns_to_plot1 = ['Total Loss', 'MSE_IC Loss']

# Plot each column's data with small markers
for column in sampled_df.columns:
    if column == 'Total Loss':  # Skip the iteration count column
        plt.plot(sampled_df.index, sampled_df[column], linewidth=2.5, label='PINN', alpha=0.9,
                color='#FF6B6B', marker='o', markevery=1, markersize=8)  # 'o' denotes a circle marker       
    if column == 'MSE_IC Loss': 
        plt.plot(sampled_df.index, sampled_df[column], linewidth=2.5, label='SDW-PINN', alpha=0.9,
                color='#4ECDC4', linestyle='--', marker='s', markevery=1, markersize=8)  # 'o' denotes a circle marker

# Use scientific notation on the y axis
plt.yscale('log')  # Use a logarithmic scale
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4],labels=['1E-04','1E-03', '1E-02', '1E-01','1E+00', '1E+01', '1E+02', '1E+03', '1E+04'],fontsize=10)  # Set ticks and labels manually

# Format the x axis as 0, 1k, 2k
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*0.1)}k' if x != 0 else '0'))

# Add labels and legend
plt.xlabel('Iteration',  fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.title('Case 3', fontsize=16)  # Chart title
# plt.legend()  # Display the legend
# Configure the legend separately
legend = plt.legend(fontsize=13)
# plt.grid(True)  # Display the grid
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/Case3/Case3.png',dpi=600)  # Display the chart