import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Create legend entries manually

# Load the data
file_path = '/home/users/liyifan/LYF/PINN_applications/Parameterization/Q3/Comsol-Label_values_with_predictions.xlsx'
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
    
    x_grid, y_grid = np.meshgrid(np.linspace(np.min(x), np.max(x), 101), np.linspace(np.min(y), np.max(y), 101))

    plt.figure(figsize=(8, 8))
    
    # Plot reference contours as solid blue lines
    contour_true = plt.contour(x_grid, y_grid, true_values, colors='blue', linestyles='-', linewidths=3)
    plt.clabel(contour_true, inline=True, fontsize=8, fmt='%.2f')
    
    # Plot predicted contours as dashed red lines
    contour_pred = plt.contour(x_grid, y_grid, pred_values, colors='red', linestyles='--', linewidths=3)
    plt.clabel(contour_pred, inline=True, fontsize=8, fmt='%.2f')
    
    # cbar = plt.colorbar(label=r'$h/m$')
    # cbar.ax.tick_params(labelsize=19)
    # cbar.set_label(r'$H/m$', fontsize=19, fontweight='bold')
    # plt.xlabel(r'$x/m$', fontsize=19, fontweight='bold')
    # plt.ylabel(r'$y/m$', fontsize=19, fontweight='bold')
    # plt.tick_params(axis='both', which='major', labelsize=19, width=1.5)
    # plt.title(f'Contour Plot at {time}')
    plt.xlabel(r'$x/m$', fontsize=22, fontweight='bold')
    plt.ylabel(r'$y/m$', fontsize=22, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=22, width=2)  # Thicken the ticks
    plt.gca().spines['top'].set_linewidth(2)  # Thicken the top border
    plt.gca().spines['right'].set_linewidth(2)  # Thicken the right border
    plt.gca().spines['left'].set_linewidth(2)  # Thicken the left border
    plt.gca().spines['bottom'].set_linewidth(2)  # Thicken the bottom border
    
    # Create legend entries manually
    legend_elements = [Line2D([0], [0], color='blue', lw=3, label='Simulator'),
                       Line2D([0], [0], color='red', lw=3, linestyle='--', label='SDW-PINN')]
    # plt.legend(handles=legend_elements, loc='best')
    plt.legend(handles=legend_elements, loc='best', fontsize=19)
    
    plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Parameterization/Q3/result/Contour_{time_label}.png')
    plt.close()
