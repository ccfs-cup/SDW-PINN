import pandas as pd

# Read the Excel file, assuming the filename is 'data.xlsx'
file_path = '/home/users/liyifan/LYF/PINN_applications/Parameterization/Q4/Comsol-Label450.xlsx'
df = pd.read_excel(file_path)

# Initialize an empty list to store results
sorted_data = []

# Process batches of 101 rows
batch_size = 101
for i in range(0, len(df), batch_size):
    # Extract the 101 rows in the current batch
    batch = df.iloc[i:i + batch_size]
    
    # Sort by the second column (assumed to be 'Column2') in ascending order
    sorted_batch = batch.sort_values(by=batch.columns[1], ascending=True)
    
    # Append the sorted results to the list
    sorted_data.append(sorted_batch)

# Combine all sorted batches
sorted_df = pd.concat(sorted_data)

# Save the sorted data to a new Excel file
sorted_df.to_excel('/home/users/liyifan/LYF/PINN_applications/Parameterization/Q4/Comsol-Label450-sorted.xlsx', index=False)

print("数据已按 101 行分批排序，并保存到 'sorted_output.xlsx'")