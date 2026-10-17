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
pinn_l2 = [0.01394135, 0.01929878, 0.02026802, 0.02018483, 0.02013898, 0.02080654, 0.02202146, 0.02377752, 0.02559195, 0.02730756, 0.02887305, 0.03011461, 0.03113489, 0.0317923, 0.03216159, 0.03217528, 0.03179656, 0.03097788, 0.02972296, 0.02819705]
pinn_r2 = [0.58627612, 0.73973912, 0.83700826, 0.88906249, 0.91617598, 0.92789331, 0.93202214, 0.93123262, 0.92912394, 0.92679158, 0.92470574, 0.92370508, 0.92336617, 0.92428199, 0.92610683, 0.92908341, 0.93328342, 0.93876177, 0.94529187, 0.95208338]
sdw_pinn_l2 = [0.00249129, 0.00130439, 0.00241246, 0.0039036, 0.00438423, 0.00395186, 0.00323233, 0.00240581, 0.00191127, 0.00164358, 0.0014834, 0.00140658, 0.00132614, 0.00127943, 0.00114534, 0.00094061, 0.00074313, 0.00066827, 0.00061667, 0.00076388]
sdw_pinn_r2 = [0.98678859, 0.99881104, 0.99769079, 0.99585086, 0.99602735, 0.99739877, 0.99853544, 0.999296, 0.99960469, 0.9997348, 0.99980126, 0.99983355, 0.99986097, 0.99987737, 0.99990629, 0.99993939, 0.99996356, 0.9999715, 0.99997645, 0.99996483]
# Create the chart
fig, ax1 = plt.subplots(figsize=(8, 6))

# # Plot L² error on the left axis
# ax1.plot(time_steps, pinn_l2, 'b-o', label='PINN L² Error', linewidth=1.5, markersize=6)
# ax1.plot(time_steps, sdw_pinn_l2, 'r-s', label='SDW-PINN L² Error', linewidth=1.5, markersize=6)
# ax1.set_xlabel("Time Steps", fontsize=12)
# ax1.set_ylabel("Relative L² Error", fontsize=12, color='darkred')
# ax1.tick_params(axis='y', labelcolor='darkred')
# ax1.grid(True, linestyle="--", alpha=0.6)

# # Create the right axis for R² scores
# ax2 = ax1.twinx()
# ax2.plot(time_steps, pinn_r2, 'g-^', label='PINN R² Score', linewidth=1.5, markersize=6)
# ax2.plot(time_steps, sdw_pinn_r2, 'm-v', label='SDW-PINN R² Score', linewidth=1.5, markersize=6)
# ax2.set_ylabel("R² Score", fontsize=12, color='darkblue')
# ax2.tick_params(axis='y', labelcolor='darkblue')
# Plot L² error on the left axis
ax1.plot(time_steps, pinn_l2, 'k-o', label='PINN L² Error', linewidth=1.5, markersize=6)
ax1.plot(time_steps, sdw_pinn_l2, 'k-s', label='SDW-PINN L² Error', linewidth=1.5, markersize=6)
ax1.set_xlabel("Time Steps", fontsize=12)
ax1.set_ylabel("Relative L² Error", fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle="--", alpha=0.6)

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
plt.title("Non-uniform rainfall recharge", fontsize=14)

# Adjust the layout and save the image
fig.tight_layout()
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case2_linear_rainfall_recharge/specified_data_plot.png", dpi=300)
