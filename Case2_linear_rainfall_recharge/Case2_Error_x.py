import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

neu_num = 60  # Number of neurons in each hidden layer

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
        self.output_layer = nn.Linear(neu_num, 1)

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], axis=1)
        layer1_out = self.act(self.hidden_layer1(inputs))
        layer2_out = self.act(self.hidden_layer2(layer1_out))
        layer3_out = self.act(self.hidden_layer3(layer2_out))
        layer4_out = self.act(self.hidden_layer4(layer3_out))
        layer5_out = self.act(self.hidden_layer5(layer4_out))
        layer6_out = self.act(self.hidden_layer6(layer5_out))
        output = self.output_layer(layer6_out)
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load pretrained weights
    pretrained_model_path = "/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/pth/final.pth"
    PINN = Net().to(device)
    state_dict = torch.load(pretrained_model_path)

    # Remove the 'module.' prefix
    state_dict = remove_module_prefix(state_dict)
    PINN.load_state_dict(state_dict)
    PINN.eval()

    # Read the Excel data
    file_path = '/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/Comsol-Label2.xlsx'
    data = pd.read_excel(file_path)

    # Extract x and y coordinates and time columns
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    time_columns = data.columns[2:]

    x_target = 50  # Position of the cross-section at x=50
    y_line = np.arange(0, 101)  # y axis from 0 to 100

    predictions = []
    errors = []

    for time_col in time_columns:
        t = float(time_col.split('=')[1])
        t_tensor = torch.tensor([t], dtype=torch.float32).repeat(len(y), 1).to(device)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)

        with torch.no_grad():
            pred_values = PINN(x_tensor, y_tensor, t_tensor).cpu().numpy().flatten()
            predictions.append(pred_values)
            data[f'Prediction_{time_col}'] = pred_values
            data[f'Error_{time_col}'] = np.abs(data[time_col].values - pred_values)

    output_file_path = '/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/Comsol-Label_values_with_predictions.xlsx'
    data.to_excel(output_file_path, index=False)

    print(f"模型预测值和误差值已保存至 {output_file_path}")

    times_to_plot = [5, 10, 15, 20]
    plt.figure(figsize=(8, 6))
    for idx, time_col in enumerate(times_to_plot):
        time_str = f't={time_col}'
        true_values = data.loc[np.isclose(data.iloc[:, 0], x_target), time_str].values
        pred_values = data.loc[np.isclose(data.iloc[:, 0], x_target), f'Prediction_{time_str}'].values

        # Calculate mAE, RMSE, and R²
        mAE = mean_absolute_error(true_values.flatten(), pred_values.flatten())
        RMSE = np.sqrt(mean_squared_error(true_values.flatten(), pred_values.flatten()))
        R2 = r2_score(true_values.flatten(), pred_values.flatten())

        print(f"Time: {time_str}")
        print(f"Mean Absolute Error (mAE): {mAE:.5f}")
        print(f"Root Mean Squared Error (RMSE): {RMSE:.5f}")
        print(f"R-squared (R²): {R2:.5f}\n")

        # Set labels only for the first time step
        if idx == 0:
            plt.plot(y_line, true_values, label='Reference', color='blue', linestyle='-', linewidth=3)
            plt.plot(y_line, pred_values, label='SDW-PINN', color='red', linestyle='--', linewidth=3)
        else:
            plt.plot(y_line, true_values, color='blue', linestyle='-', linewidth=3)
            plt.plot(y_line, pred_values, color='red', linestyle='--', linewidth=3)

    plt.xlabel(r'$y/m$', fontsize=12, fontweight='bold')  # Display using LaTeX formatting
    plt.ylabel(r'$H/m$', fontsize=12, fontweight='bold')  # Display using LaTeX formatting
    plt.legend(fontsize=12, frameon=True, framealpha=1, edgecolor="black", loc="best")  # Thicken the legend border
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)  # Thicken the ticks
    plt.gca().spines['top'].set_linewidth(1.5)  # Thicken the top border
    plt.gca().spines['right'].set_linewidth(1.5)  # Thicken the right border
    plt.gca().spines['left'].set_linewidth(1.5)  # Thicken the left border
    plt.gca().spines['bottom'].set_linewidth(1.5)  # Thicken the bottom border
    plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/result/Line_Plot_x_50.png')
    plt.close()


if __name__ == "__main__":
    main()
