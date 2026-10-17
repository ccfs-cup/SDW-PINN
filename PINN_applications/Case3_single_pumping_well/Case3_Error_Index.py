# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D  # Create legend entries manually
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load the data
# file_path = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Label_values_with_predictions.xlsx'
# data = pd.read_excel(file_path)

# # Extract x, y, and predicted and reference values at each time point
# x = data.iloc[:, 0].values
# y = data.iloc[:, 1].values

# time_columns = [col for col in data.columns if 'Prediction_' in col]
# times_to_plot = ['t=5', 't=10', 't=15', 't=20']  # Time points to plot

# # Generate contour plots
# for time in times_to_plot:
#     time_label = f't={time.split("=")[-1]}'
#     true_values = data[time_label].values.reshape(101, 101).T
#     pred_values = data[f'Prediction_{time_label}'].values.reshape(101, 101).T
#     # Calculate mAE, RMSE, and R²
#     mAE = mean_absolute_error(true_values.flatten(), pred_values.flatten())
#     RMSE = np.sqrt(mean_squared_error(true_values.flatten(), pred_values.flatten()))
#     R2 = r2_score(true_values.flatten(), pred_values.flatten())
    
#     print(f"Time: {time}")
#     print(f"Mean Absolute Error (mAE): {mAE:.5f}")
#     print(f"Root Mean Squared Error (RMSE): {RMSE:.5f}")
#     print(f"R-squared (R²): {R2:.5f}\n")
    
#     x_grid, y_grid = np.meshgrid(np.linspace(np.min(x), np.max(x), 101), np.linspace(np.min(y), np.max(y), 101))

#     plt.figure(figsize=(8, 8))
    
#     # Plot reference contours as solid blue lines
#     contour_true = plt.contour(x_grid, y_grid, true_values, colors='blue', linestyles='-', linewidths=3)
#     plt.clabel(contour_true, inline=True, fontsize=8, fmt='%.2f')
    
#     # Plot predicted contours as dashed red lines
#     contour_pred = plt.contour(x_grid, y_grid, pred_values, colors='red', linestyles='--', linewidths=3)
#     plt.clabel(contour_pred, inline=True, fontsize=8, fmt='%.2f')
    
#     plt.title(f'Contour Plot at {time}')
#     plt.xlabel(r'$x/m$', fontsize=15, fontweight='bold')
#     plt.ylabel(r'$y/m$', fontsize=15, fontweight='bold')
#     plt.tick_params(axis='both', which='major', labelsize=12, width=2)  # Thicken the ticks
#     plt.gca().spines['top'].set_linewidth(1.5)  # Thicken the top border
#     plt.gca().spines['right'].set_linewidth(1.5)  # Thicken the right border
#     plt.gca().spines['left'].set_linewidth(1.5)  # Thicken the left border
#     plt.gca().spines['bottom'].set_linewidth(1.5)  # Thicken the bottom border
    
#     # Create legend entries manually
#     legend_elements = [Line2D([0], [0], color='blue', lw=3, label='COMSOL'),
#                        Line2D([0], [0], color='red', lw=3, linestyle='--', label='SDW-PINN')]
#     plt.legend(handles=legend_elements, loc='best')
    
#     plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/result/Contour_{time_label}.png')
#     plt.close()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Create legend entries manually
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = '/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/Comsol-Label_values_with_predictions.xlsx'
data = pd.read_excel(file_path)

# Extract x, y, and predicted and reference values at each time point
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

time_columns = [col for col in data.columns if 'Prediction_' in col]
times_to_plot = ['t=5', 't=10', 't=15', 't=20']  # Time points to plot

# Generate contour plots
for time in times_to_plot:
    time_label = f't={time.split("=")[-1]}'
    true_values = data[time_label].values.reshape(101, 101).T
    pred_values = data[f'Prediction_{time_label}'].values.reshape(101, 101).T
    
    # Calculate the relative L2 error and R²
    relative_L2_error = np.linalg.norm(true_values.flatten() - pred_values.flatten()) / np.linalg.norm(true_values.flatten())
    R2 = r2_score(true_values.flatten(), pred_values.flatten())
    
    print(f"Time: {time}")
    print(f"Relative L2 Error: {relative_L2_error:.5f}")
    print(f"R-squared (R²): {R2:.5f}\n")
    
    x_grid, y_grid = np.meshgrid(np.linspace(np.min(x), np.max(x), 101), np.linspace(np.min(y), np.max(y), 101))

    plt.figure(figsize=(8, 8))
    
    # Plot reference contours as solid blue lines
    contour_true = plt.contour(x_grid, y_grid, true_values, colors='blue', linestyles='-', linewidths=3)
    plt.clabel(contour_true, inline=True, fontsize=8, fmt='%.2f')
    
    # Plot predicted contours as dashed red lines
    contour_pred = plt.contour(x_grid, y_grid, pred_values, colors='red', linestyles='--', linewidths=3)
    plt.clabel(contour_pred, inline=True, fontsize=8, fmt='%.2f')
    
    plt.title(f'Contour Plot at {time}')
    plt.xlabel(r'$x/m$', fontsize=15, fontweight='bold')
    plt.ylabel(r'$y/m$', fontsize=15, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)  # Thicken the ticks
    plt.gca().spines['top'].set_linewidth(1.5)  # Thicken the top border
    plt.gca().spines['right'].set_linewidth(1.5)  # Thicken the right border
    plt.gca().spines['left'].set_linewidth(1.5)  # Thicken the left border
    plt.gca().spines['bottom'].set_linewidth(1.5)  # Thicken the bottom border
    
    # Create legend entries manually
    legend_elements = [Line2D([0], [0], color='blue', lw=3, label='Reference'),
                       Line2D([0], [0], color='red', lw=3, linestyle='--', label='SDW-PINN')]
    plt.legend(handles=legend_elements, loc='best')
    
    # plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/result/Contour_{time_label}.png')
    plt.close()
