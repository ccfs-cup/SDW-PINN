import pandas as pd

# Read the TXT file and convert it to a DataFrame
file_path = '/home/users/liyifan/LYF/PINN_applications/Parameterization/Q4/450.txt'
df = pd.read_csv(file_path, delim_whitespace=True)

# Save as an Excel file
output_file = '/home/users/liyifan/LYF/PINN_applications/Parameterization/Q4/Comsol-Label450.xlsx'
df.to_excel(output_file, index=False)

print(f"文件已成功保存为: {output_file}")
