import pandas as pd

# Read the TXT file and convert it to a DataFrame
file_path = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.1/comsol-label-0.1.txt'
df = pd.read_csv(file_path, delim_whitespace=True)

# Save as an Excel file
output_file = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/M0.1/comsol-label-0.1.xlsx'
df.to_excel(output_file, index=False)

print(f"File saved successfully to: {output_file}")
