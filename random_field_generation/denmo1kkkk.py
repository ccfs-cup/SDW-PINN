import pandas as pd

# Read the TXT file
data = pd.read_csv('PINN/random_field_generation/k_values_3.txt', delimiter='\t', header=None)

# Write to an Excel file
output_file = 'PINN/random_field_generation/Original_field.xlsx'
data.to_excel(output_file, index=False, header=False)

print(f'The data has been successfully written to {output_file}')
