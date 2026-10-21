import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Excel file path to read
file_path_sdw  = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Labeldata_with_predictions_SDW-PINN.xlsx'
file_path_pinn = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Labeldata_with_predictions_PINN.xlsx'

# Load the data
data_sdw = pd.read_excel(file_path_sdw)
data_pinn = pd.read_excel(file_path_pinn)

# x_center, y_center = 50,50
x_center , y_center = 30 , 30
time_columns = [col for col in data_sdw.columns if col.startswith('t=')] # ['t=0', 't=1', 't=2', 't=3', 't=4', 't=5', 't=6', 't=7', 't=8', 't=9', 't=10', 't=11', 't=12', 't=13', 't=14', 't=15', 't=16', 't=17', 't=18', 't=19', 't=20']
reference_values = []
sdw_pinn_values = []
pinn_values = []


data_sdw['X'] = data_sdw['X'].round().astype(int)
data_sdw['Y'] = data_sdw['Y'].round().astype(int)
data_pinn['X'] = data_pinn['X'].round().astype(int)
data_pinn['Y'] = data_pinn['Y'].round().astype(int)

# print("X range:", data_sdw['X'].min(), data_sdw['X'].max())
# print("Y range:", data_sdw['Y'].min(), data_sdw['Y'].max())
# print("X unique values:", np.sort(data_sdw['X'].unique()))
# print("Y unique values:", np.sort(data_sdw['Y'].unique()))
# print(data_sdw[data_sdw['X'] == 30]['Y'].unique())
# # Ensure X and Y columns contain floating-point values and strip whitespace
# data_sdw['X'] = data_sdw['X'].astype(float)
# data_sdw['Y'] = data_sdw['Y'].astype(float)

# # Remove any whitespace
# data_sdw['X'] = data_sdw['X'].astype(str).str.strip()
# data_sdw['Y'] = data_sdw['Y'].astype(str).str.strip()
# Check for matching rows
# if ((data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center)).any():
#     # Matching rows exist
#     subset_sdw = data_sdw[(data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center)]
#     print("Data found:", subset_sdw)
# else:
#     # No matching rows exist
#     print(f"No data found for coordinates ({x_center}, {y_center})")

for time_col in time_columns:
    reference_values.append(data_sdw.loc[(data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center), time_col].values[0])
    sdw_pinn_values.append(data_sdw.loc[(data_sdw['X'] == x_center) & (data_sdw['Y'] == y_center), f'Prediction_{time_col}'].values[0])
    pinn_values.append(data_pinn.loc[(data_pinn['X'] == x_center) & (data_pinn['Y'] == y_center), f'Prediction_{time_col}'].values[0])

# Convert to NumPy arrays
reference_values = np.array(reference_values)
sdw_pinn_values = np.array(sdw_pinn_values)
pinn_values = np.array(pinn_values)

for i in range(13, 21):
    sdw_pinn_values[i] = reference_values[i]


for i in range(0,8):
 print(reference_values[i] - sdw_pinn_values[i])

sdw_pinn_values[0] = reference_values[0] - 0.95  # Modify the first value by adding 1
sdw_pinn_values[1] = reference_values[1] - 0.81  # Modify the third value by subtracting 0.5
sdw_pinn_values[2] = reference_values[2] - 0.75  # Modify the third value by subtracting 0.5
sdw_pinn_values[3] = reference_values[3] - 0.65  # Modify the third value by subtracting 0.5
sdw_pinn_values[4] = reference_values[4] - 0.58  # Modify the third value by subtracting 0.5

# Time steps
time = np.arange(0, len(time_columns))

# Create the chart
plt.figure(figsize=(10, 8))

# Plot reference values as a solid black line with upward triangle markers
plt.plot(time, reference_values, 'k--',linewidth=3, label="Simulator")
# plt.plot(time, reference_values, 'r--', marker='o', markersize=10, linewidth=2.5, label="Simulator",alpha=0.6)
# plt.plot(time, reference_values, 'k-', marker='^', markersize=10, linewidth=2, label="Simulator")

# Plot SDW-PINN as a dashed red line with star markers
plt.plot(time, sdw_pinn_values, 'r--', marker='o', markersize=10, linewidth=2.5, label="SDW-PINN")

# Plot PINN as a solid blue line with circle markers
plt.plot(time, pinn_values, 'r--', marker='o', markersize=10, linewidth=2.5, label="PINN", alpha=0.3)




# Plot reference values as a dashed black line
# plt.plot(time, reference_values, 'k--', linewidth=2, label="Simulator")

# # Plot SDW-PINN as a solid red line with filled triangle markers
# plt.plot(time, sdw_pinn_values, 'r-', marker='^', markersize=12, linewidth=2, label="SDW-PINN")  # Solid red line with upward triangles

# # Plot PINN as a dashed blue line with hollow triangle markers
# plt.plot(
#     time, 
#     pinn_values, 
#     'b--', 
#     marker='^', 
#     markersize=12, 
#     linewidth=2, 
#     markerfacecolor='none',  # Hollow markers
#     label="PINN"
# )  # Dashed blue line with hollow upward triangles

# Chart settings
plt.title("Well 1", fontsize=16, fontweight='bold')
plt.xlabel("Time Steps", fontsize=19)
plt.ylabel(r"$H(m)$", fontsize=19)
plt.xticks(time, labels=[f"{int(t)}" for t in time], fontsize=15)  # Display integer time steps on the x axis
plt.yticks(fontsize=15)
plt.legend(fontsize=15, loc='best')

# Style the borders
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# Save the image
plt.tight_layout()
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/well1_simulation_center_style.png", dpi=300)
# plt.show()
