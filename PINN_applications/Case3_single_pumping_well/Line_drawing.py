import matplotlib.pyplot as plt
import numpy as np

# Specify numerical data
time_steps = np.arange(1, 21)  # Time steps (1 to 20)
# 8000
# pinn_l2 = [0.00340435, 0.00274167, 0.00181749, 0.00179617, 0.0018874, 0.00206224, 0.00242122, 0.00302557, 0.00355999, 0.00401837, 0.0044398, 0.00475732, 0.00508834, 0.00533072, 0.00556346, 0.00572921, 0.00579192, 0.00568987, 0.00541211, 0.00511764]
# pinn_r2 = [0.9140097, 0.98676655, 0.99731176, 0.99842078, 0.99879507, 0.99891971, 0.99881398, 0.99846344, 0.99817572, 0.9979567, 0.99776363, 0.99766065, 0.99753089, 0.9974722, 0.99740917, 0.99739515, 0.99746026, 0.99764926, 0.99795079, 0.99822767]
# # 5000
# pinn_l2 = [0.00748324, 0.00848873, 0.00654747, 0.00461531, 0.00346544, 0.0034089, 0.00405713, 0.00531868, 0.00669908, 0.00801778, 0.00923925, 0.01020166, 0.0110417, 0.01164817, 0.01212935, 0.01242035, 0.01243116, 0.01200576, 0.01101874, 0.00959103]
# pinn_r2 = [0.8807988, 0.94964593, 0.98299054, 0.99419999, 0.99751795, 0.99806444, 0.99769265, 0.99655922, 0.99514351, 0.99368893, 0.99229007, 0.99124447, 0.99036177, 0.98983586, 0.98948996, 0.98943253, 0.98980243, 0.99080188, 0.99248149, 0.99445618]


# 4000
pinn_l2 = [0.01299092, 0.02006595, 0.02413364, 0.02727194, 0.02977206, 0.03145678, 0.03354354, 0.03506564, 0.03665963, 0.03762471, 0.03864913, 0.03928349, 0.039649, 0.03972843, 0.03946154, 0.03888181, 0.0379874, 0.03687557, 0.03579212, 0.03566523]
pinn_r2 = [0.92190972, 0.87434085, 0.85075296, 0.83426033, 0.82192324, 0.81554237, 0.80455635, 0.79816352, 0.7909499, 0.78936022, 0.78715442, 0.78831394, 0.79166886, 0.79729687, 0.80559493, 0.81610484, 0.82854753, 0.84190094, 0.85395667, 0.85758472]
sdw_pinn_l2 = [0.00291942, 0.00159951, 0.00226746, 0.00191686, 0.00144926, 0.00153839, 0.0011114, 0.0010854, 0.00076874, 0.00079641, 0.00058221, 0.00055773, 0.00062144, 0.00073468, 0.00084811, 0.00097088, 0.00107325, 0.00118002, 0.00125583, 0.00132479]
sdw_pinn_r2 = [0.99605623, 0.99920155, 0.99868253, 0.99918121, 0.99957803, 0.99955883, 0.99978544, 0.99980662, 0.99990807, 0.99990562, 0.9999517, 0.99995733, 0.99994882, 0.99993068, 0.9999102, 0.99988534, 0.99986314, 0.99983811, 0.99982021, 0.9998035]
# Create the chart
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot L² error on the left axis
ax1.plot(time_steps, pinn_l2, 'k-o', label='PINN L² Error', linewidth=1.5, markersize=6)
ax1.plot(time_steps, sdw_pinn_l2, 'k-s', label='SDW-PINN L² Error', linewidth=1.5, markersize=6)
ax1.set_xlabel("Time Steps", fontsize=12)
ax1.set_ylabel("Relative L² Error", fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle="--", alpha=0.6)

# # Create the right axis for R² scores
# ax2 = ax1.twinx()
# ax2.plot(time_steps, pinn_r2, 'g-^', label='PINN R² Score', linewidth=1.5, markersize=6)
# ax2.plot(time_steps, sdw_pinn_r2, 'm-v', label='SDW-PINN R² Score', linewidth=1.5, markersize=6)
# ax2.set_ylabel("R² Score", fontsize=12, color='darkblue')
# ax2.tick_params(axis='y', labelcolor='darkblue')
# Create the right axis for R² scores
ax2 = ax1.twinx()
ax2.plot(time_steps, pinn_r2, 'r-o', label='PINN R² Score', linewidth=1.5, markersize=6)
ax2.plot(time_steps, sdw_pinn_r2, 'r-s', label='SDW-PINN R² Score', linewidth=1.5, markersize=6)
ax2.set_ylabel("R² Score", fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Set integer step sizes on the x axis
ax1.set_xticks(time_steps)  # Ensure the x axis displays integers

# # Add a legend in an empty area of the plot
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.05, 0.9), fontsize=10)
# Add the legend below the chart
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10, ncol=2)

# Add a title
plt.title("Single pumping well", fontsize=14)

# Adjust the layout and save the image
fig.tight_layout()
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case3_single_pumping_well/specified_data_plot.png", dpi=300)
