import numpy as np
import matplotlib.pyplot as plt

# Define the grid range
x = np.linspace(0, 100, 100)  # x direction: 0 to 100 m
y = np.linspace(0, 100, 100)  # y direction: 0 to 100 m
X, Y = np.meshgrid(x, y)

# Define the recharge intensity S(x, y, t)
S = 5e-4 * X  # Recharge intensity increases with x

# Plot the nonuniform recharge intensity distribution
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, S, cmap='jet', levels=100)
# plt.colorbar(contour, label='$S(x, y, t)$ (m/d)')
cbar = plt.colorbar(label=r'$S(x, y, t)$ (m/d)')
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r'$(m/d)$', fontsize=12)
plt.xlabel(r'$x/m$', fontsize=19, fontweight='bold')
plt.ylabel(r'$y/m$', fontsize=19, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=19, width=1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.title('Non-uniform Recharge Strength Distribution')
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/123.png',dpi = 300)
