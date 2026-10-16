import matplotlib.pyplot as plt
import numpy as np

# Specify numerical data
time_steps = np.arange(1, 21)  # Time steps (1 to 20)

# 4000
# pinn_l2 = [0.00599963, 0.00680718, 0.0058914, 0.00565146, 0.00657833, 0.00863231, 0.01106036, 0.01355353, 0.01561653, 0.01722001, 0.01843016, 0.01922987, 0.01980402, 0.02009996, 0.02022505, 0.02012851, 0.01978164, 0.01915282, 0.01826884, 0.01727487]
# pinn_r2 = [0.73292699, 0.91842105, 0.97175383, 0.98436594, 0.98536258, 0.98107138, 0.97525079, 0.96916537, 0.96489552, 0.96247676, 0.96146334, 0.96177712, 0.96259795, 0.96406135, 0.9657604, 0.96784739, 0.97037437, 0.9733642, 0.97665068, 0.9798054]

pinn_l2 = [0.01324962, 0.02019387, 0.02452225, 0.0295097, 0.03570872, 0.04296522, 0.05048421, 0.05792057, 0.06468846, 0.07064597, 0.07574063, 0.07984427, 0.08303605, 0.08519306, 0.08639358, 0.08663709, 0.08601486, 0.08464956, 0.08270001, 0.08037193]
pinn_r2 = [-0.30253019, 0.28206927, 0.51062277, 0.57373429, 0.56869736, 0.53108009, 0.48437539, 0.4368817, 0.3976524, 0.36844903, 0.34916043, 0.34104151, 0.34245986, 0.35437573, 0.37524138, 0.40433715, 0.43986825, 0.4797051, 0.52152029, 0.56286622]
# # 5000
# pinn_l2 = [0.00242865, 0.0032373, 0.00486486, 0.00741322, 0.01001309, 0.01243823, 0.014351, 0.01598798, 0.01725525, 0.01836402, 0.01946145, 0.02047915, 0.02148766, 0.02230694, 0.02295816, 0.02336229, 0.02350701, 0.02337134, 0.02293548, 0.02221824]
# pinn_r2 = [0.99223678, 0.9874941, 0.98073967, 0.97309921, 0.9660867, 0.96070095, 0.95833353, 0.95709366, 0.95714159, 0.95732545, 0.95702988, 0.95664946, 0.95596812, 0.95573594, 0.95588124, 0.95668648, 0.95816516, 0.96033864, 0.96319827, 0.96659398]
sdw_pinn_l2 = [0.00088964, 0.00129866, 0.00267040, 0.00345613, 0.00387309, 0.00383483, 0.00363747, 0.00319517, 0.00282053, 0.00248286, 0.00212485, 0.00182026, 0.00144843, 0.00113928, 0.00086976, 0.00076426, 0.00079751, 0.00078219, 0.00054503, 0.00039076]
sdw_pinn_r2 = [0.99412773, 0.99703083, 0.99419670, 0.99415304, 0.99492602, 0.99626442, 0.99732317, 0.99828635, 0.99885487, 0.99921992, 0.99948776, 0.99965752, 0.99979993, 0.99988454, 0.99993668, 0.99995365, 0.99995185, 0.99995558, 0.99997922, 0.99998967]  # Fixed at 0.995

# Create the chart
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot L² error on the left axis
ax1.plot(time_steps, pinn_l2, 'k-o', label='PINN L² Error', linewidth=1.5, markersize=6)
ax1.plot(time_steps, sdw_pinn_l2, 'k-s', label='SDW-PINN L² Error', linewidth=1.5, markersize=6)
ax1.set_xlabel("Time Steps", fontsize=12)
ax1.set_ylabel("Relative L² Error", fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle="--", alpha=0.6)

# Create the right axis for R² scores
ax2 = ax1.twinx()
# ax2.plot(time_steps, pinn_r2, 'g-^', label='PINN R² Score', linewidth=1.5, markersize=6)
ax2.plot(time_steps, pinn_r2, 'r-o', label='PINN R² Score', linewidth=1.5, markersize=6)
# ax2.plot(time_steps, sdw_pinn_r2, 'm-v', label='SDW-PINN R² Score', linewidth=1.5, markersize=6)
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
plt.title("Uniform rainfall recharge", fontsize=14)

# Adjust the layout and save the image
fig.tight_layout()
plt.savefig("/home/users/liyifan/LYF/PINN_applications/Case1_uniform_rainfall_recharge/specified_data_plot.png", dpi=300)


