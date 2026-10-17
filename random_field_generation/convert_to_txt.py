import pandas as pd

def excel_to_txt(input_excel, output_txt):
    # Read the Excel file
    df = pd.read_excel(input_excel, header=None)
    # Transpose the data
    df = df.T

    # Open the output file
    with open(output_txt, 'w') as f:
        # Write the header
        f.write("x\ty\tvalue\n")

        # Iterate over each cell and format the output
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                f.write(f"{i}\t{j}\t{df.iloc[i, j]}\n")

# Usage example
input_excel = '/home/users/liyifan/LYF/PINN/random_field_generation/Result/new_high_quality_k_fields/permeability_field_lognormal_5.xlsx'  # Input Excel file path
output_txt = '/home/users/liyifan/LYF/PINN/random_field_generation/Result/new_high_quality_k_fields/K_field_lognormal_5.txt'  # Output TXT file path

excel_to_txt(input_excel, output_txt)
print(f"文件已成功保存为 {output_txt}")
