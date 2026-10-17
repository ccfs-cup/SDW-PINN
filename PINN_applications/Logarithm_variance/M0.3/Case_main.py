import scipy.io as scio
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.autograd import Variable
import seaborn as sns
import os
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

neu_num = 120  # Number of neurons in each hidden layer
# num = 14000  # Number of observations

n_c = 100    # Number of PDE collocation points
n_ic1 = 80000  # Number of initial condition 1 points (entire domain)
n_ic2 = 5000   # Number of initial condition 2 points (left and right boundaries)
n_bc  = 5000   # Left and right Dirichlet boundary conditions
# n_ic1 = 20000  # Number of initial condition 1 points (entire domain)
# n_ic2 = 2000   # Number of initial condition 2 points (left and right boundaries)
# n_bc  = 2000   # Left and right Dirichlet boundary conditions

lr = 1e-3  # Initial learning rate
step_size = 4000  # Learning rate decay step size
gamma = 0.5  # Learning rate decay factor
# iterations = 30000  # Number of training iterations
iterations = 20000  # Number of training iterations

# # k_path = r'E:\Desktop\data\3D\Flield\permeability_fields_3D.mat'
# k_path = r'/home/cc/CCFs/LYF/PINN/3D/permeability_fields_3D.mat'
# k = scio.loadmat(k_path)['K']

# print("k shape:", k.shape)  # Expected shape: (10, 10, 10)
# Read permeability K values from the file
Ksx_data = np.loadtxt('/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.3.txt')
# Convert to a PyTorch tensor
Ksx_tensor = torch.tensor(Ksx_data, dtype=torch.float32)[:, 2].view(101, 101)

Ksy_data = np.loadtxt('/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.3.txt')
# Convert to a PyTorch tensor
Ksy_tensor = torch.tensor(Ksy_data, dtype=torch.float32)[:, 2].view(101, 101)

# # Define finite differences in the x direction
# def diff_x(u, dx):
#     u_l = u[:, :-1, :]
#     u_r = u[:, 1:, :]
#     diff_u_x = (u_r - u_l) / dx
#     # Add a dimension to preserve the original shape
#     diff_u_x = torch.cat((diff_u_x, diff_u_x[:, -1:, :]), dim=1)
#     return diff_u_x

# # Define finite differences in the y direction
# def diff_y(u, dy):
#     u_u = u[:-1, :, :]
#     u_l = u[1:, :, :]
#     diff_u_y = (u_l - u_u) / dy
#     # Add a dimension to preserve the original shape
#     diff_u_y = torch.cat((diff_u_y, diff_u_y[-1:, :, :]), dim=0)
#     return diff_u_y
# Define the finite difference function in the y direction
def diff_x(u, dy):
    """Compute finite differences in the y direction."""
    du_dy = torch.zeros_like(u)
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)  # Use central differences
    du_dy[0, :] = (u[1, :] - u[0, :]) / dy  # Use forward differences at the lower boundary
    du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy  # Use backward differences at the upper boundary
    return du_dy

def diff_y(u, dx):
    """Compute finite differences in the x direction."""
    du_dx = torch.zeros_like(u)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)  # Use central differences
    du_dx[:, 0] = (u[:, 1] - u[:, 0]) / dx  # Use forward differences at the left boundary
    du_dx[:, -1] = (u[:, -1] - u[:, -2]) / dx  # Use backward differences at the right boundary
    return du_dx



# # Define finite differences in the z direction
# def diff_z(u, dz):
#     u_f = u[:, :, :-1]
#     u_b = u[:, :, 1:]
#     diff_u_z = (u_b - u_f) / dz
#     # Add a dimension to preserve the original shape
#     diff_u_z = torch.cat((diff_u_z, diff_u_z[:, :, -1:]), dim=2)
#     return diff_u_z

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
# Feedforward network with six hidden layers using Swish activation and a final linear output layer
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
        inputs = torch.cat([x, y, t], axis=1)  # combined two arrays of 1 columns each to one array of 3 columns
        layer1_out = self.act(self.hidden_layer1(inputs))
        layer2_out = self.act(self.hidden_layer2(layer1_out))
        layer3_out = self.act(self.hidden_layer3(layer2_out))
        layer4_out = self.act(self.hidden_layer4(layer3_out))
        layer5_out = self.act(self.hidden_layer5(layer4_out))
        layer6_out = self.act(self.hidden_layer6(layer5_out))
        layer7_out = self.act(self.hidden_layer6(layer6_out))
        layer8_out = self.act(self.hidden_layer6(layer7_out))
        output = self.output_layer(layer8_out)  # For regression, no activation is used in output layer
        return output

net = Net()

# Check the number of available GPUs
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个 GPU！")
    # Use DataParallel to run the model on multiple GPUs
    # net = nn.DataParallel(net)
    net = nn.DataParallel(net, device_ids=[1,2,3,4,6,7])
    
net = net.to(device)

# # Load model parameters and remove the 'module.' prefix
# def remove_module_prefix(state_dict):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # Remove the 'module.' prefix if present in the parameter name
#         if k.startswith("module."):
#             new_state_dict[k[len("module."):]] = v
#         else:
#             new_state_dict[k] = v
#     return new_state_dict

# # Load pretrained weights
# pretrained_model_path = "/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/pth/high_quality_weights/final.pth"
# if os.path.exists(pretrained_model_path):
#     # # Load the pretrained model state_dict
#     # state_dict = torch.load(pretrained_model_path)

#     # # Remove the 'module.' prefix
#     # state_dict = remove_module_prefix(state_dict)
#     # net.load_state_dict(state_dict)

#     net.load_state_dict(torch.load(pretrained_model_path))
#     print(f"Loaded pretrained model weights from {pretrained_model_path}")
# else:
#     print(f"Pretrained model weights not found at {pretrained_model_path}")
    
    
mse_cost_function = torch.nn.MSELoss()  # Mean squared error         #Average the squared differences between model predictions and actual targets
optimizer = torch.optim.Adam(net.parameters(), lr)  # Optimize with Adam at learning rate lr, updating net.parameters()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)  # Adjust the learning rate dynamically
# Convert to a PyTorch tensor
# k = torch.tensor(k, dtype=torch.float)

# dx, dy, dz = 1, 1, 1
# diff_k_x = diff_x(k, dx)
# diff_k_y = diff_y(k, dy)
# diff_k_z = diff_z(k, dz)
# Calculate permeability K differences in the x and y directions
# Assume dx and dy are the grid spacings
dx, dy = 1, 1  # Assumed grid spacing
Ksx_diff_x = diff_x(Ksx_tensor, dx)
Ksy_diff_y = diff_y(Ksy_tensor, dy)
# # Print the tensor shapes after differencing
# print("diff_k_x shape:", diff_k_x.shape)
# print("diff_k_y shape:", diff_k_y.shape)
# print("diff_k_z shape:", diff_k_z.shape)


def f(func_input, net):
    x =     func_input[:, 0].reshape(-1, 1).float().to(device)
    y =     func_input[:, 1].reshape(-1, 1).float().to(device)
    t =     func_input[:, 2].reshape(-1, 1).float().to(device)
    ksx =   func_input[:, 3].reshape(-1, 1).float().to(device)
    ksx_x = func_input[:, 4].reshape(-1, 1).float().to(device)
    ksy =   func_input[:, 5].reshape(-1, 1).float().to(device)
    ksy_y = func_input[:, 6].reshape(-1, 1).float().to(device)

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True
    u = net(x, y, t)
    u_x  = torch.autograd.grad(u.sum(),   x, retain_graph=True, create_graph=True, allow_unused=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, retain_graph=True, create_graph=True)[0]
    u_y  = torch.autograd.grad(u.sum(),   y, retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, retain_graph=True, create_graph=True)[0]
    u_t  = torch.autograd.grad(u.sum(), t, retain_graph=True, create_graph=True)[0]
    # source_term1 =  500 / (2 * torch.pi * 2**2) * torch.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 2**2))
    # source_term2 =  600 / (2 * torch.pi * 2**2) * torch.exp(-((x - 50)**2 + (y - 80)**2) / (2 * 2**2))
    # source_term3 =  800 / (2 * torch.pi * 2**2) * torch.exp(-((x - 70)**2 + (y - 50)**2) / (2 * 2**2))
    source_term1 =  500 / (2 * torch.pi * 2**2) * torch.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 2**2))
    source_term2 =  600 / (2 * torch.pi * 2**2) * torch.exp(-((x - 30)**2 + (y - 70)**2) / (2 * 2**2))
    source_term3 =  700 / (2 * torch.pi * 2**2) * torch.exp(-((x - 70)**2 + (y - 50)**2) / (2 * 2**2))

    
    pde = 0.1 * u_t - ksx * u_xx - ksy * u_yy - ksx_x * u_x - ksy_y * u_y + source_term1 + source_term2 + source_term3
    return pde


## Data from initial Conditions
# Initial condition 1: n_ic1 = 80000
x_ic1 = np.random.uniform(low=0.0, high=100.0, size=(n_ic1, 1)) # Generate uniformly distributed random numbers throughout the spatial domain
y_ic1 = np.random.uniform(low=0.0, high=100.0, size=(n_ic1, 1))
t_ic1 = np.zeros((n_ic1, 1))
u_ic1 = np.ones((n_ic1, 1)) * 100
# Initial condition 2: n_ic2 = 5000
x_ic2 = np.zeros((n_ic2, 1))
y_ic2 = np.random.uniform(low=0.0, high=100.0, size=(n_ic2, 1))
t_ic2 = np.zeros((n_ic2, 1))
u_ic2 = np.ones((n_ic2, 1)) * 100
# Initial condition 3: n_ic3 = 5000
x_ic3 = np.ones((n_ic2, 1)) * 100
y_ic3 = np.random.uniform(low=0.0, high=100.0, size=(n_ic2, 1))
t_ic3 = np.zeros((n_ic2, 1))
u_ic3 = np.ones((n_ic2, 1)) * 100

x_ic = np.vstack((x_ic1, x_ic2, x_ic3))
y_ic = np.vstack((y_ic1, y_ic2, y_ic3))
t_ic = np.vstack((t_ic1, t_ic2, t_ic3))
u_ic = np.vstack((u_ic1, u_ic2, u_ic3))
ic = np.hstack((x_ic, y_ic, t_ic, u_ic))
np.random.shuffle(ic) # Use NumPy's shuffle function to randomly reorder the rows of ic
x_ic, y_ic, t_ic, u_ic = ic[:, 0].reshape(-1, 1), ic[:, 1].reshape(-1, 1), ic[:, 2].reshape(-1, 1), ic[:, 3].reshape(-1,1)


## Data from Boundary Conditions 
# n_bc  = 5000
# Constant-head left boundary u=5 (x=0.0)
x_bc1 = np.zeros((n_bc, 1))
y_bc1 = np.random.uniform(low=0.0, high=100.0, size=(n_bc, 1))
t_bc1 = np.random.uniform(low=0.0, high=20.0,  size=(n_bc, 1))
u_bc1 = np.ones((n_bc, 1)) * 100
# Constant-head right boundary u=5 (x=100)
x_bc2 = np.ones((n_bc, 1)) * 100
y_bc2 = np.random.uniform(low=0.0, high=100.0, size=(n_bc, 1))
t_bc2 = np.random.uniform(low=0.0, high=20.0,  size=(n_bc, 1))
u_bc2 = np.ones((n_bc, 1)) * 100

x_bc = np.vstack((x_bc1, x_bc2))
y_bc = np.vstack((y_bc1, y_bc2))
t_bc = np.vstack((t_bc1, t_bc2))
u_bc = np.vstack((u_bc1, u_bc2))
bc = np.hstack((x_bc, y_bc, t_bc, u_bc))
np.random.shuffle(bc)
x_bc, y_bc , t_bc, u_bc = bc[:, 0].reshape(-1, 1), bc[:, 1].reshape(-1, 1), bc[:, 2].reshape(-1, 1), bc[:, 3].reshape(-1,1)  # Extract columns from bc and reshape them into four column vectors

## Data from Boundary Conditions 
n_nf  = 10000
# No-flow upper boundary (y=100.0)
x_nf1 = np.random.uniform(low=0.0, high=100.0, size=(n_nf, 1))
y_nf1 = np.ones((n_nf, 1)) * 100
t_nf1 = np.random.uniform(low=0.0, high=20.0, size=(n_nf, 1))
u_nf1 = np.zeros((n_nf, 1))
# No-flow lower boundary (y=0)
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
x_nf, y_nf , t_nf, u_nf = nf[:, 0].reshape(-1, 1), nf[:, 1].reshape(-1, 1), nf[:, 2].reshape(-1, 1), nf[:, 3].reshape(-1,1)  # Extract columns from bc and reshape them into four column vectors


## Data from collection points
# x_collocation = np.random.uniform(low=0.0, high=100.0, size=(n_c, 1))
# y_collocation = np.random.uniform(low=0.0, high=100.0, size=(n_c, 1))
x_collocation = np.linspace(1.0, 99.0, 99).reshape(99, 1)
y_collocation = np.linspace(1.0, 99.0, 99).reshape(99, 1)
t_collocation = np.linspace(1.0, 20.0, 20).reshape(20, 1)

# Loss based on PDE
pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)

# Get the number of collocation points
num_x = pt_x_collocation.shape[0] # num_x = 100
num_y = pt_y_collocation.shape[0]
num_t = pt_t_collocation.shape[0]


func_input = []

# Extract the collocation point coordinates for the current iteration
for i in range(num_x):
    for j in range(num_y):
        for t in range(num_t):
            x_value = pt_x_collocation[i, 0].cpu().detach().numpy()
            y_value = pt_y_collocation[j, 0].cpu().detach().numpy()
            t_value = pt_t_collocation[t, 0].cpu().detach().numpy()

            index_x = int(x_value)
            index_y = int(y_value)
            # index_z = int(x_value - 0.5)

            # # Ensure the indices are within valid bounds
            # index_x = max(0, min(index_x, 9))
            # index_y = max(0, min(index_y, 9))
            # print(index_y,index_y)
            # # Get the values in k at the corresponding indices

            ksx_value = Ksx_tensor[index_x, index_y].detach().numpy()
            ksy_value = Ksy_tensor[index_x, index_y].detach().numpy()
            ksx_x = Ksx_diff_x[index_x, index_y].detach().numpy()
            ksy_y = Ksy_diff_y[index_x, index_y].detach().numpy()
            # k_z = diff_k_z[index_x, index_y].detach().numpy()

            # func_input.append([x_value, y_value, z_value,t_value, k_value, k_x, k_y,k_z])
            func_input.append([x_value, y_value,t_value, ksx_value, ksx_x, ksy_value , ksy_y])



func_input = torch.from_numpy(np.vstack(func_input))
all_zeros = np.zeros((n_c, 1))

# # Loss based on DATA
# Read the Excel file
file_path1 = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/comsol-label-0.3-sorted.xlsx'
data_label = pd.read_excel(file_path1)

# Extract (x, y) coordinates and time-step data
x = data_label.iloc[:, 0].values
y = data_label.iloc[:, 1].values
time_steps = data_label.columns[2:]

# Initialize an array to store (x, y, t, u)
total_rows = len(x) * len(time_steps)
p_t = np.zeros((total_rows, 4))

# Populate the array
row_index = 0
for t_index, time_step in enumerate(time_steps):
    u = data_label[time_step].values
    for i in range(len(x)):
        p_t[row_index] = [x[i], y[i], t_index, u[i]]
        row_index += 1

# Shuffle the array
np.random.shuffle(p_t)

# Select half of the data at each time step
sampled_p_t = []
for t_index in range(len(time_steps)):
    t_mask = p_t[:, 2] == t_index
    sampled_data = p_t[t_mask]
    sampled_indices = np.random.choice(len(sampled_data), int(len(sampled_data) // 1.5), replace=False)
    sampled_p_t.append(sampled_data[sampled_indices])

# Combine the sampled arrays
sampled_p_t = np.vstack(sampled_p_t)

x_t = sampled_p_t[:,0].reshape(-1,1)
y_t = sampled_p_t[:,1].reshape(-1,1)
t_t = sampled_p_t[:,2].reshape(-1,1)
u_t = sampled_p_t[:,3].reshape(-1,1)

# Initialize lists to store loss values
total_loss_list = []
mse_d_loss_list = []
# mse_f_loss_list = []
# mse_bc_loss_list = []
# mse_ic_loss_list = []
# mse_nf_loss_list = []


for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero

    # Loss based on train data
    pt_x_t = Variable(torch.from_numpy(x_t).float(), requires_grad=False).to(device)
    pt_y_t = Variable(torch.from_numpy(y_t).float(), requires_grad=False).to(device)
    pt_t_t = Variable(torch.from_numpy(t_t).float(), requires_grad=False).to(device)
    pt_u_t = Variable(torch.from_numpy(u_t).float(), requires_grad=False).to(device)
    net_d_out = net(pt_x_t, pt_y_t ,pt_t_t)  
    mse_d = mse_cost_function(net_d_out, pt_u_t)

    # # Loss based on PDE
    # pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    # f_out = f(func_input, net)
    # mse_f = mse_cost_function(f_out, torch.zeros_like(f_out))

    # # Loss based on boundary conditions
    # pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    # pt_y_bc = Variable(torch.from_numpy(y_bc).float(), requires_grad=False).to(device)
    # pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    # pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)

    # net_bc_out = net(pt_x_bc, pt_y_bc, pt_t_bc)  # output of u(x,t)
    # mse_bc = mse_cost_function(net_bc_out, pt_u_bc)

    # # Loss based on initial conditions
    # pt_x_ic = Variable(torch.from_numpy(x_ic).float(), requires_grad=False).to(device)
    # pt_y_ic = Variable(torch.from_numpy(y_ic).float(), requires_grad=False).to(device)
    # pt_t_ic = Variable(torch.from_numpy(t_ic).float(), requires_grad=False).to(device)
    # pt_u_ic = Variable(torch.from_numpy(u_ic).float(), requires_grad=False).to(device)

    # net_ic_out = net(pt_x_ic, pt_y_ic, pt_t_ic)  # output of u(x,t)
    # mse_ic = mse_cost_function(net_ic_out, pt_u_ic)
    
    # # noflow conditions
    # pt_x_nf = Variable(torch.from_numpy(x_nf).float(), requires_grad=True).to(device)
    # pt_y_nf = Variable(torch.from_numpy(y_nf).float(), requires_grad=True).to(device)
    # pt_t_nf = Variable(torch.from_numpy(t_nf).float(), requires_grad=True).to(device)
    # pt_u_nf = Variable(torch.from_numpy(u_nf).float(), requires_grad=True).to(device)
    # net_nf_out = net(pt_x_nf, pt_y_nf, pt_t_nf)  # output of u(x,t)
    # u_y = torch.autograd.grad(net_nf_out.sum(), pt_y_nf, retain_graph=True, create_graph=True)[0]
    # mse_nf = mse_cost_function(u_y, pt_u_nf)




    # loss =  10 * mse_d + 100 * mse_f + mse_bc + mse_ic + 10 * mse_nf
    # loss =  100 * mse_f +  mse_bc +  mse_ic + 8 * mse_nf
    loss =  mse_d
    # loss = mse_t + 1000 * mse_f + mse_bc + mse_ic
    loss.backward() 
    optimizer.step()
    scheduler.step() 
    
  
    # Record the loss values for each epoch
    total_loss_list.append(loss.item())
    mse_d_loss_list.append(mse_d.item())
    # mse_f_loss_list.append(mse_f.item())
    # mse_bc_loss_list.append(mse_bc.item())
    # mse_ic_loss_list.append(mse_ic.item())
    # mse_nf_loss_list.append(mse_nf.item())
    
    # Get the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch % 100 == 0: 
        with torch.no_grad():
            # print(f"Epoch {epoch}/{iterations} - Training Loss: {loss.item():.5e}, dloss: {mse_d.data:.5e} , floss: {mse_f.data:.5e}, bcloss: {mse_bc.data:.5e}, icloss: {mse_ic.data:.5e} , nfloss: {mse_nf.data:.5e}")
            # print(f"Epoch {epoch}/{iterations} - Training Loss: {loss.item():.5e}, floss: {mse_f.data:.5e}, bcloss: {mse_bc.data:.5e}, icloss: {mse_ic.data:.5e} , nfloss: {mse_nf.data:.5e}")
            print(f"Epoch {epoch}/{iterations} - Training Loss: {loss.item():.5e}, floss: {mse_d.data:.5e}")
            print(f"Current Learning Rate: {current_lr:.5e}")
            
    # Save the model every 1000 epochs
    if epoch % 1000 == 0:
        save_path = r"/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/pth/final_epoch_{}.pth".format(epoch)
        torch.save(net.state_dict(), save_path)  # Save the model parameters
        # print(f"Model saved to: {save_path}")

# Save the loss values to an Excel file
loss_data = {
    'Total Loss': total_loss_list,
    'MSE_D Loss': mse_d_loss_list,
    # 'MSE_F Loss': mse_f_loss_list,
    # 'MSE_BC Loss': mse_bc_loss_list,
    # 'MSE_IC Loss': mse_ic_loss_list,
    # 'MSE_NF Loss': mse_nf_loss_list
}

df = pd.DataFrame(loss_data)
df.to_excel('/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/loss/loss_values2.xlsx', index=False)

model_path = r"/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/pth/final.pth"
torch.save(net.state_dict(), model_path)  # Save only the model parameters

print("模型已成功保存到：", model_path)

# Plot the loss curves
plt.figure(figsize=(10, 8))
plt.plot(total_loss_list, label='Total Loss')
plt.plot(mse_d_loss_list, label='MSE_D Loss')
# plt.plot(mse_f_loss_list, label='MSE_F Loss')
# plt.plot(mse_bc_loss_list, label='MSE_BC Loss')
# plt.plot(mse_ic_loss_list, label='MSE_IC Loss')
# plt.plot(mse_nf_loss_list, label='MSE_IC Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
# Set the y axis to a logarithmic scale
plt.yscale('log')
plt.ylim(1e-4, 10)
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.3/loss/loss_curve2.png")

