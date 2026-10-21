import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Create legend entries manually
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)
GRID_SIZE = 101


def reshape_grid(values):
    return values.reshape(GRID_SIZE, GRID_SIZE).T


def plot_3d_contour(x_grid, y_grid, true_values, pred_values, output_path, time_label):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    contour_true = ax.contour3D(
        x_grid,
        y_grid,
        true_values,
        levels=18,
        colors="blue",
        linewidths=2.2,
    )
    contour_pred = ax.contour3D(
        x_grid,
        y_grid,
        pred_values,
        levels=18,
        colors="red",
        linestyles="--",
        linewidths=2.2,
    )
    ax.clabel(contour_true, inline=True, fontsize=7, fmt="%.2f")
    ax.clabel(contour_pred, inline=True, fontsize=7, fmt="%.2f")

    ax.set_title(f"3D Contour {time_label}", fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel(r"$x/m$", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"$y/m$", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_zlabel(r"$H/m$", fontsize=15, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="z", which="major", labelsize=12)
    ax.view_init(elev=28, azim=-135)
    ax.set_box_aspect((1, 1, 0.45))

    legend_elements = [
        Line2D([0], [0], color="blue", lw=3, label="Simulator"),
        Line2D([0], [0], color="red", lw=3, linestyle="--", label="SDW-PINN"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

# Load the data
file_path = BASE_DIR / "Comsol-Label_values_with_predictions.xlsx"
data = pd.read_excel(file_path)

# Extract x, y, and predicted and reference values at each time point
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
x_grid = reshape_grid(x)
y_grid = reshape_grid(y)

time_columns = [col for col in data.columns if 'Prediction_' in col]
times_to_plot = ['t=5', 't=10', 't=15', 't=20']  # Time points to plot

# Generate contour plots
for time in times_to_plot:
    time_label = f't={time.split("=")[-1]}'
    true_values = reshape_grid(data[time_label].values)
    pred_values = reshape_grid(data[f'Prediction_{time_label}'].values)
    plot_3d_contour(
        x_grid,
        y_grid,
        true_values,
        pred_values,
        RESULT_DIR / f'Contour_3D_{time_label}.png',
        time_label,
    )
