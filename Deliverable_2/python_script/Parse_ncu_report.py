#Parse code for unifying ncu tests in test/GPU_test_{sys.argv[1]}.csv

import sys
import pandas as pd
import os
import glob
import sys

# Folder containing your CSV files
csv_folder = 'test/report_ncu'

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))

# Read and combine all CSV files
df_list = []
for file in csv_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV file
combined_df.to_csv(f'test/combined_ncu_report_{sys.argv[1]}.csv', index=False)
