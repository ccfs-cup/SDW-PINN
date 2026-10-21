import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

neu_num = 120  # Number of neurons in each hidden layer

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
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # Load pretrained weights
    pretrained_model_path = "/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/pth/final_epoch_10000.pth"
    # pretrained_model_path = "/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/pth/final.pth"
    
    PINN = Net().to(device)
    # Load the pretrained model state_dict
    state_dict = torch.load(pretrained_model_path)

    # Remove the 'module.' prefix
    state_dict = remove_module_prefix(state_dict)
    PINN.load_state_dict(state_dict)
    PINN.eval()

    # Read the Excel data
    file_path = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/comsol-label-0.3-sorted.xlsx'
    data = pd.read_excel(file_path)

    # Extract x and y coordinates and time columns
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    time_columns = data.columns[2:]

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

    output_file_path = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/comsol-label-0.3-sorted-with-predictions.xlsx'
    data.to_excel(output_file_path, index=False)

    print(f"Model predictions and errors saved to {output_file_path}")


    # Plot the figure
    for time_col in time_columns:
        pred_values = data[f'Prediction_{time_col}'].values.reshape(101, 101).T
        true_values = data[time_col].values.reshape(101, 101).T
        error_values = data[f'Error_{time_col}'].values.reshape(101, 101).T

        plt.figure(figsize=(8, 6))
        plt.imshow(true_values, extent=(0, 100, 0, 100), origin='lower', cmap='jet', vmin=44.7, vmax=100.0)
        cbar = plt.colorbar(label=r'$h/m$')
        cbar.ax.tick_params(labelsize=19)
        cbar.set_label(r'$H/m$', fontsize=19, fontweight='bold')
        plt.xlabel(r'$x/m$', fontsize=19, fontweight='bold')
        plt.ylabel(r'$y/m$', fontsize=19, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=19, width=1.5)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/result/Label_{time_col}.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.imshow(pred_values, extent=(0, 100, 0, 100), origin='lower', cmap='jet', vmin=44.7, vmax=100.0)
        cbar = plt.colorbar(label=r'$h/m$')
        cbar.ax.tick_params(labelsize=19)
        cbar.set_label(r'$H/m$', fontsize=19, fontweight='bold')
        plt.xlabel(r'$x/m$', fontsize=19, fontweight='bold')
        plt.ylabel(r'$y/m$', fontsize=19, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=19, width=1.5)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/result/Predicted_{time_col}.png')
        plt.close()

        plt.figure(figsize=(8, 6))
        # plt.imshow(error_values, extent=(0, 100, 0, 100), origin='lower', cmap='jet', vmin=0.0, vmax=1.3)
        plt.imshow(error_values, extent=(0, 100, 0, 100), origin='lower', cmap='jet')
        cbar = plt.colorbar(label=r'$h/m$')
        cbar.ax.tick_params(labelsize=19)
        cbar.set_label(r'$H/m$', fontsize=19, fontweight='bold')
        plt.xlabel(r'$x/m$', fontsize=19, fontweight='bold')
        plt.ylabel(r'$y/m$', fontsize=19, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=19, width=1.5)
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        plt.savefig(f'/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/result/Error_{time_col}.png')
        plt.close()


if __name__ == "__main__":
    main()

