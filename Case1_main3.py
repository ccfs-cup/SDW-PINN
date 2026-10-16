import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.autograd import Variable
import seaborn as sns
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===================== Hyperparameters =====================
neu_num = 60  # Number of neurons in each hidden layer

n_c = 100    # Number of PDE points (variable name retained; these are grid collocation points)
n_ic1 = 80000  # Number of initial condition 1 points (entire domain)
n_ic2 = 5000   # Number of initial condition 2 points (left and right boundaries)
n_bc  = 5000   # Left and right Dirichlet boundary conditions

lr = 1e-3  # Initial network learning rate
step_size = 4000  # Learning rate decay step size
gamma = 0.5  # Learning rate decay factor
iterations = 30000

# Dynamic weight hyperparameters
lr_lambda = 1e-4   # Learning rate for loss weights
lambda_min = 0.1   # Lower weight bound to prevent excessively small weights
lambda_max = 1000.0 # Upper weight bound to prevent numerical explosion

# ===================== Read the permeability field =====================
K_data = np.loadtxt('/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/K1_noisy.txt')
# Convert to a PyTorch tensor of shape (101, 101), with dimension 0 along x and dimension 1 along y
K_tensor = torch.tensor(K_data, dtype=torch.float32)[:, 2].view(101, 101)
dx, dy = 1, 1  # Grid spacing

# ===================== 1. Compute permeability gradients using Scharr convolution =====================
def scharr_gradient(K, dx, dy):
    """
    Compute permeability field gradients in the x/y directions using Scharr convolution.
    Matching formula: S_x = 1/(32*2Δx) * kernel matrix; use replicate padding for boundary accuracy.
    Input: K (Nx, Ny), dx, dy grid spacings.
    Output: Kx=∂K/∂x, Ky=∂K/∂y.
    """
    # Convert to the 4D format required for convolution (batch, channel, H, W)
    K_4d = K.unsqueeze(0).unsqueeze(0)

    # ∂K/∂x: differences along dimension 0 (rows), using the vertical Scharr kernel (top-bottom differences)
    kernel_x = torch.tensor([
        [-3, -10, -3],
        [0,   0,   0],
        [3,  10,   3]
    ], dtype=torch.float32)
    coeff_x = 1.0 / (32.0 * 2.0 * dx)  # Normalization factor and spatial step scaling
    kernel_x = kernel_x * coeff_x
    kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)

    # ∂K/∂y: differences along dimension 1 (columns), using the horizontal Scharr kernel (left-right differences)
    kernel_y = torch.tensor([
        [-3,  0,  3],
        [-10, 0, 10],
        [-3,  0,  3]
    ], dtype=torch.float32)
    coeff_y = 1.0 / (32.0 * 2.0 * dy)
    kernel_y = kernel_y * coeff_y
    kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)

    # Replicate boundary padding to avoid gradient distortion caused by zero padding
    pad = nn.ReplicationPad2d(1)
    K_pad = pad(K_4d)

    # Compute convolutions and squeeze dimensions
    Kx = F.conv2d(K_pad, kernel_x, stride=1, padding=0).squeeze(0).squeeze(0)
    Ky = F.conv2d(K_pad, kernel_y, stride=1, padding=0).squeeze(0).squeeze(0)

    return Kx, Ky

# Calculate permeability gradients
K_diff_x, K_diff_y = scharr_gradient(K_tensor, dx, dy)

# ===================== 2. Xavier initialization =====================
def xavier_init(m):
    """Apply Xavier uniform initialization to fully connected layers and initialize biases to 0."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Define the activation function
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

# Define the neural network
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

net = Net()
# Apply Xavier initialization
net.apply(xavier_init)

# Multi-GPU parallelism
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU！")
    net = nn.DataParallel(net)
    
net = net.to(device)

# ===================== Optimizer and loss =====================
mse_cost_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

# ===================== PDE residual function =====================
def f(func_input, net):
    x = func_input[:, 0].reshape(-1, 1).float().to(device)
    y = func_input[:, 1].reshape(-1, 1).float().to(device)
    t = func_input[:, 2].reshape(-1, 1).float().to(device)
    k = func_input[:, 3].reshape(-1, 1).float().to(device)
    kx = func_input[:, 4].reshape(-1, 1).float().to(device)
    ky = func_input[:, 5].reshape(-1, 1).float().to(device)

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    u = net(x, y, t)
    
    u_x  = torch.autograd.grad(u.sum(),   x, retain_graph=True, create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, retain_graph=True, create_graph=True)[0]
    u_y  = torch.autograd.grad(u.sum(),   y, retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, retain_graph=True, create_graph=True)[0]
    u_t  = torch.autograd.grad(u.sum(), t, retain_graph=True, create_graph=True)[0]
    
    pde = 0.1 * u_t - k * (u_xx + u_yy) - kx * u_x - ky * u_y - 0.025
    return pde

# ===================== Initial condition data =====================
x_ic1 = np.random.uniform(low=0.0, high=100.0, size=(n_ic1, 1))
y_ic1 = np.random.uniform(low=0.0, high=100.0, size=(n_ic1, 1))
t_ic1 = np.zeros((n_ic1, 1))
u_ic1 = np.ones((n_ic1, 1)) * 5

x_ic2 = np.zeros((n_ic2, 1))
y_ic2 = np.random.uniform(low=0.0, high=100.0, size=(n_ic2, 1))
t_ic2 = np.zeros((n_ic2, 1))
u_ic2 = np.ones((n_ic2, 1)) * 5

x_ic3 = np.ones((n_ic2, 1)) * 100
y_ic3 = np.random.uniform(low=0.0, high=100.0, size=(n_ic2, 1))
t_ic3 = np.zeros((n_ic2, 1))
u_ic3 = np.ones((n_ic2, 1)) * 5

x_ic = np.vstack((x_ic1, x_ic2, x_ic3))
y_ic = np.vstack((y_ic1, y_ic2, y_ic3))
t_ic = np.vstack((t_ic1, t_ic2, t_ic3))
u_ic = np.vstack((u_ic1, u_ic2, u_ic3))
ic = np.hstack((x_ic, y_ic, t_ic, u_ic))
np.random.shuffle(ic)
x_ic, y_ic, t_ic, u_ic = ic[:, 0].reshape(-1, 1), ic[:, 1].reshape(-1, 1), ic[:, 2].reshape(-1, 1), ic[:, 3].reshape(-1,1)

# ===================== Dirichlet boundary conditions =====================
x_bc1 = np.zeros((n_bc, 1))
y_bc1 = np.random.uniform(low=0.0, high=100.0, size=(n_bc, 1))
t_bc1 = np.random.uniform(low=0.0, high=20.0,  size=(n_bc, 1))
u_bc1 = np.ones((n_bc, 1)) * 5

x_bc2 = np.ones((n_bc, 1)) * 100
y_bc2 = np.random.uniform(low=0.0, high=100.0, size=(n_bc, 1))
t_bc2 = np.random.uniform(low=0.0, high=20.0,  size=(n_bc, 1))
u_bc2 = np.ones((n_bc, 1)) * 5

x_bc = np.vstack((x_bc1, x_bc2))
y_bc = np.vstack((y_bc1, y_bc2))
t_bc = np.vstack((t_bc1, t_bc2))
u_bc = np.vstack((u_bc1, u_bc2))
bc = np.hstack((x_bc, y_bc, t_bc, u_bc))
np.random.shuffle(bc)
x_bc, y_bc , t_bc, u_bc = bc[:, 0].reshape(-1, 1), bc[:, 1].reshape(-1, 1), bc[:, 2].reshape(-1, 1), bc[:, 3].reshape(-1,1)

# ===================== No-flow boundary conditions =====================
n_nf  = 10000
x_nf1 = np.random.uniform(low=0.0, high=100.0, size=(n_nf, 1))
y_nf1 = np.ones((n_nf, 1)) * 100
t_nf1 = np.random.uniform(low=0.0, high=20.0, size=(n_nf, 1))
u_nf1 = np.zeros((n_nf, 1))

x_nf2 = np.random.uniform(low=0.0, high=100.0, size=(n_nf, 1))
y_nf2 = np.zeros((n_nf, 1))
t_nf2 = np.random.uniform(low=0.0, high=20.0, size=(n_nf, 1))
u_nf2 = np.zeros((n_nf, 1))

x_nf = np.vstack((x_nf1, x_nf2))
y_nf = np.vstack((y_nf1, y_nf2))
t_nf = np.vstack((t_nf1, t_nf2))
u_nf = np.vstack((u_nf1, u_nf2))
nf = np.hstack((x_nf, y_nf, t_nf, u_nf))
np.random.shuffle(nf)
x_nf, y_nf , t_nf, u_nf = nf[:, 0].reshape(-1, 1), nf[:, 1].reshape(-1, 1), nf[:, 2].reshape(-1, 1), nf[:, 3].reshape(-1,1)

# ===================== PDE collocation points (vectorized to replace three nested loops) =====================
x_collocation = np.linspace(1.0, 99.0, 99).reshape(99, 1)
y_collocation = np.linspace(1.0, 99.0, 99).reshape(99, 1)
t_collocation = np.linspace(1.0, 20.0, 20).reshape(20, 1)

# Generate and flatten a 3D grid
x_mesh, y_mesh, t_mesh = np.meshgrid(
    x_collocation.ravel(), 
    y_collocation.ravel(), 
    t_collocation.ravel(), 
    indexing='ij'
)
x_flat = x_mesh.ravel()
y_flat = y_mesh.ravel()
t_flat = t_mesh.ravel()

# Calculate grid indices and extract permeability and gradient values
index_x = np.clip(x_flat.astype(int), 0, 100)
index_y = np.clip(y_flat.astype(int), 0, 100)

k_values = K_tensor[index_x, index_y].numpy()
kx_values = K_diff_x[index_x, index_y].numpy()
ky_values = K_diff_y[index_x, index_y].numpy()

# Concatenate into the collocation input tensor [x, y, t, k, kx, ky]
func_input = torch.tensor(
    np.column_stack([x_flat, y_flat, t_flat, k_values, kx_values, ky_values]),
    dtype=torch.float32
)

# ===================== Training loop =====================
total_loss_list = []
mse_f_loss_list = []
mse_bc_loss_list = []
mse_ic_loss_list = []
mse_nf_loss_list = []

# 3. Initialize dynamic loss weights for PDE, BC, IC, and NF
lambda_f = 100.0
lambda_bc = 1.0
lambda_ic = 1.0
lambda_nf = 8.0

for epoch in range(iterations):
    optimizer.zero_grad()

    # ---------- 1. PDE loss ----------
    f_out = f(func_input, net)
    mse_f = mse_cost_function(f_out, torch.zeros_like(f_out))

    # ---------- 2. Boundary loss ----------
    pt_x_bc = torch.from_numpy(x_bc).float().to(device)
    pt_y_bc = torch.from_numpy(y_bc).float().to(device)
    pt_t_bc = torch.from_numpy(t_bc).float().to(device)
    pt_u_bc = torch.from_numpy(u_bc).float().to(device)
    net_bc_out = net(pt_x_bc, pt_y_bc, pt_t_bc)
    mse_bc = mse_cost_function(net_bc_out, pt_u_bc)

    # ---------- 3. Initial condition loss ----------
    pt_x_ic = torch.from_numpy(x_ic).float().to(device)
    pt_y_ic = torch.from_numpy(y_ic).float().to(device)
    pt_t_ic = torch.from_numpy(t_ic).float().to(device)
    pt_u_ic = torch.from_numpy(u_ic).float().to(device)
    net_ic_out = net(pt_x_ic, pt_y_ic, pt_t_ic)
    mse_ic = mse_cost_function(net_ic_out, pt_u_ic)
    
    # ---------- 4. No-flow boundary loss ----------
    pt_x_nf = torch.from_numpy(x_nf).float().requires_grad_(True).to(device)
    pt_y_nf = torch.from_numpy(y_nf).float().requires_grad_(True).to(device)
    pt_t_nf = torch.from_numpy(t_nf).float().to(device)
    pt_u_nf = torch.from_numpy(u_nf).float().to(device)
    net_nf_out = net(pt_x_nf, pt_y_nf, pt_t_nf)
    u_y = torch.autograd.grad(net_nf_out.sum(), pt_y_nf, retain_graph=True, create_graph=True)[0]
    mse_nf = mse_cost_function(u_y, pt_u_nf)

    # ---------- Weighted total loss ----------
    loss = lambda_f * mse_f + lambda_bc * mse_bc + lambda_ic * mse_ic + lambda_nf * mse_nf

    # Backpropagate and update network parameters (gradient descent)
    loss.backward() 
    optimizer.step()
    scheduler.step() 

    # ---------- Update loss weights dynamically (gradient ascent, following equations 3.11-3.14) ----------
    lambda_f += lr_lambda * mse_f.item()
    lambda_bc += lr_lambda * mse_bc.item()
    lambda_ic += lr_lambda * mse_ic.item()
    lambda_nf += lr_lambda * mse_nf.item()

    # Clip weights to prevent numerical explosion or vanishing weights
    lambda_f = max(lambda_min, min(lambda_f, lambda_max))
    lambda_bc = max(lambda_min, min(lambda_bc, lambda_max))
    lambda_ic = max(lambda_min, min(lambda_ic, lambda_max))
    lambda_nf = max(lambda_min, min(lambda_nf, lambda_max))
    
    # ---------- Logging ----------
    total_loss_list.append(loss.item())
    mse_f_loss_list.append(mse_f.item())
    mse_bc_loss_list.append(mse_bc.item())
    mse_ic_loss_list.append(mse_ic.item())
    mse_nf_loss_list.append(mse_nf.item())
    
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch % 100 == 0: 
        with torch.no_grad():
            print(f"Epoch {epoch}/{iterations} - Loss: {loss.item():.5e}")
            print(f"  mse_f: {mse_f.data:.5e}, mse_bc: {mse_bc.data:.5e}, mse_ic: {mse_ic.data:.5e}, mse_nf: {mse_nf.data:.5e}")
            print(f"  λ_f: {lambda_f:.3f}, λ_bc: {lambda_bc:.3f}, λ_ic: {lambda_ic:.3f}, λ_nf: {lambda_nf:.3f}")
            print(f"  LR: {current_lr:.5e}")
            
    # Save the model every 1000 epochs
    if epoch % 1000 == 0:
        save_path = r"/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/pth/qianxiang_pth/final_epoch_{}.pth".format(epoch)
        torch.save(net.state_dict(), save_path)

# ===================== Save the final model =====================
model_path = r"/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/pth/qianxiang_pth/final.pth"
torch.save(net.state_dict(), model_path)
print("模型已成功保存到：", model_path)

# ===================== Plot loss curves =====================
plt.figure(figsize=(10, 8))
plt.plot(total_loss_list, label='Total Loss')
plt.plot(mse_f_loss_list, label='MSE_F Loss')
plt.plot(mse_bc_loss_list, label='MSE_BC Loss')
plt.plot(mse_ic_loss_list, label='MSE_IC Loss')
plt.plot(mse_nf_loss_list, label='MSE_NF Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves (Scharr Gradient + Dynamic Weight)')
plt.legend()
plt.yscale('log')
plt.ylim(1e-6, 10)
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/loss/loss_curve_scharr_dynamic.png")