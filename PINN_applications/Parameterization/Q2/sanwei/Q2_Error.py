import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

neu_num = 120  # Number of neurons in each hidden layer
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
GRID_SIZE = 101


def reshape_grid(values):
    return values.reshape(GRID_SIZE, GRID_SIZE).T


def plot_3d_surface(x_grid, y_grid, z_values, output_path, z_label, title, cmap="jet", vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    cbar = fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(z_label, fontsize=15, fontweight="bold")

    ax.set_title(title, fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel(r"$x/m$", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"$y/m$", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_zlabel(z_label, fontsize=15, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="z", which="major", labelsize=12)
    ax.view_init(elev=28, azim=-135)
    ax.set_box_aspect((1, 1, 0.45))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.act = Swish()
        self.hidden_layer1 = nn.Linear(3, neu_num)
        self.hidden_layer2 = nn.Linear(neu_num, neu_num)
        self.hidden_layer3 = nn.Linear(neu_num, neu_num)
        self.hidden_layer4 = nn.Linear(neu_num, neu_num)
        self.hidden_layer5 = nn.Linear(neu_num, neu_num)
        self.hidden_layer6 = nn.Linear(neu_num, neu_num)
        self.hidden_layer7 = nn.Linear(neu_num, neu_num)
        self.hidden_layer8 = nn.Linear(neu_num, neu_num)
        self.output_layer = nn.Linear(neu_num, 1)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], axis=1)
        layer1_out = self.act(self.hidden_layer1(inputs))
        layer2_out = self.act(self.hidden_layer2(layer1_out))
        layer3_out = self.act(self.hidden_layer3(layer2_out))
        layer4_out = self.act(self.hidden_layer4(layer3_out))
        layer5_out = self.act(self.hidden_layer5(layer4_out))
        layer6_out = self.act(self.hidden_layer6(layer5_out))
        layer7_out = self.act(self.hidden_layer6(layer6_out))
        layer8_out = self.act(self.hidden_layer6(layer7_out))
        output = self.output_layer(layer8_out)
        return output
# Load model parameters and remove the 'module.' prefix
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove the 'module.' prefix if present in the parameter name
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(exist_ok=True)

    # Load pretrained weights
    pretrained_model_path = BASE_DIR / "final.pth"
    
    PINN = Net().to(device)
    # Load the pretrained model state_dict
    state_dict = torch.load(pretrained_model_path, map_location=device)

    # Remove the 'module.' prefix
    state_dict = remove_module_prefix(state_dict)
    PINN.load_state_dict(state_dict)
    PINN.eval()

    # Read the Excel data
    file_path = BASE_DIR / "Comsol-Label3.xlsx"
    data = pd.read_excel(file_path)

    # Extract x and y coordinates and time columns
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    time_columns = data.columns[2:]
    x_grid = reshape_grid(x)
    y_grid = reshape_grid(y)

    predictions = []
    errors = []

    for time_col in time_columns:
        t = float(time_col.split('=')[1])
        t_tensor = torch.tensor([t], dtype=torch.float32).repeat(len(x), 1).to(device)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)

        with torch.no_grad():
            pred_values = PINN(x_tensor, y_tensor, t_tensor).cpu().numpy().flatten()
            predictions.append(pred_values)
            data[f'Prediction_{time_col}'] = pred_values
            data[f'Error_{time_col}'] = np.abs(data[time_col].values - pred_values)

    output_file_path = BASE_DIR / "Comsol-Label_values_with_predictions.xlsx"
    data.to_excel(output_file_path, index=False)

    print(f"模型预测值和误差值已保存至 {output_file_path}")


    # Plot the figure
    for time_col in time_columns:
        pred_values = reshape_grid(data[f'Prediction_{time_col}'].values)
        true_values = reshape_grid(data[time_col].values)
        error_values = reshape_grid(data[f'Error_{time_col}'].values)

        plot_3d_surface(
            x_grid,
            y_grid,
            true_values,
            RESULT_DIR / f'Label_3D_{time_col}.png',
            r"$H/m$",
            f"Simulator {time_col}",
            vmin=81.1,
            vmax=100.0,
        )
        plot_3d_surface(
            x_grid,
            y_grid,
            pred_values,
            RESULT_DIR / f'Predicted_3D_{time_col}.png',
            r"$H/m$",
            f"PINN {time_col}",
            vmin=81.1,
            vmax=100.0,
        )
        plot_3d_surface(
            x_grid,
            y_grid,
            error_values,
            RESULT_DIR / f'Error_3D_{time_col}.png',
            r"$|Error|/m$",
            f"Absolute Error {time_col}",
        )


if __name__ == "__main__":
    main()

