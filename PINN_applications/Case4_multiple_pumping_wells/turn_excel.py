import pandas as pd

# Read the TXT file and convert it to a DataFrame
file_path = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/comsol-label4.txt'
df = pd.read_csv(file_path, delim_whitespace=True)

# Save as an Excel file
output_file = '/home/users/liyifan/LYF/PINN_applications/Case4_multiple_pumping_wells/Comsol-Label4.xlsx'
df.to_excel(output_file, index=False)

print(f"文件已成功保存为: {output_file}")
