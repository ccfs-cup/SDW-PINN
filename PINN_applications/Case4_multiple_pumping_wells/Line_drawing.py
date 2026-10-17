import matplotlib.pyplot as plt
import numpy as np

# Specify numerical data
time_steps = np.arange(1, 21)  # Time steps (1 to 20)

# 4000
pinn_l2 = [0.00841095, 0.01146889, 0.01660103, 0.02161525, 0.02617009, 0.03015039, 0.03438228, 0.03860301, 0.04254187, 0.04592429, 0.04890542, 0.05124272, 0.05313745, 0.05440568, 0.05503985, 0.05488948, 0.05379088, 0.05164375, 0.04839829, 0.04416321]
pinn_r2 = [0.9695178, 0.95893288, 0.93053539, 0.90270824, 0.8799469, 0.86337428, 0.84592714, 0.82940602, 0.81563564, 0.80634356, 0.80010302, 0.79822903, 0.79901711, 0.80332069, 0.81089001, 0.8222853, 0.83790765, 0.85747082, 0.88009541, 0.90402195]
sdw_pinn_l2 = [0.00514189, 0.00315812, 0.00358078, 0.00418036, 0.00452726, 0.00456612, 0.0036566, 0.00231986, 0.00105765, 0.00073319, 0.00133054, 0.00182626, 0.00224112, 0.00252321, 0.00277502, 0.00299006, 0.00311906, 0.00310842, 0.00291613, 0.00268048]
sdw_pinn_r2 = [0.98860795, 0.99688606, 0.99676817, 0.99636099, 0.99640719, 0.99686642, 0.99825735, 0.99938391, 0.99988605, 0.99995064, 0.99985204, 0.99974372, 0.99964249, 0.99957696, 0.99951928, 0.99947264, 0.999455, 0.99948365, 0.9995647, 0.99964643]
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
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/specified_data_plot.png", dpi=300)
