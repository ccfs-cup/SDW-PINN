import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the plotting style
# sns.set(style="white", palette="bright")

# Read the data
df = pd.read_excel('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/loss_values .xlsx', sheet_name='Sheet1')

# Create the figure
plt.figure(figsize=(12, 8))

# Plot the two curves
plt.plot(df['Iteration'], df['SDW-PINN_Loss'], 
         linewidth=2.5, label='SDW-PINN', alpha=0.9,
         color='#FF6B6B', marker='o', markevery=1, markersize=8)

plt.plot(df['Iteration'], df['PINN_Loss'], 
         linewidth=2.5, label='PINN', alpha=0.9,
         color='#4ECDC4', linestyle='--', marker='s', markevery=1, markersize=8)

# Configure the axes
plt.xlabel('Iteration', fontsize=18, labelpad=10)
plt.ylabel('Loss', fontsize=18, labelpad=10)

# Configure the axes (key modification)
xticks_pos = np.arange(0, 10001, 1000)  # [0, 1000, 2000,...,10000]
xticks_labels = [f'{int(x/1000)}k' if x !=0 else '0' for x in xticks_pos]  # Generate labels 0, 1k, ..., 10k

plt.xticks(ticks=xticks_pos, labels=xticks_labels, fontsize=18)
# plt.xticks(np.arange(0, 10001, 1000), fontsize=10)
plt.yticks(fontsize=15)

# Configure visualization details
plt.title('Csae 1', 
         fontsize=18, pad=20, fontweight='bold')
plt.legend(loc='upper right', fontsize=18, framealpha=0.9)

# Add auxiliary elements
# plt.grid(True, alpha=0.3)
plt.xlim(0, 10000)
plt.tight_layout()

# Highlight key regions
# plt.axvspan(2000, 4000, alpha=0.1, color='gray', label='Key Convergence Phase')

# Display and save
plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/normalized_loss.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/users/liyifan/LYF/PINN_applications/Dynamic_weighting/filename.svg', format='svg')