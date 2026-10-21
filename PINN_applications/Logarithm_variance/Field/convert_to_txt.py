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
input_excel = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.3.xlsx'  # Input Excel file path
output_txt = '/home/users/liyifan/LYF/PINN_applications/Logarithm_variance/Field/permeability_field_lognormal_0.3.txt'  # Output TXT file path

excel_to_txt(input_excel, output_txt)
print(f"File saved successfully to {output_txt}")
