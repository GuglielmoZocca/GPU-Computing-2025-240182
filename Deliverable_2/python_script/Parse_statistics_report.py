import sys
import pandas as pd
import os

#Parse code for clean test information in test/combined_ncu_report_{sys.argv[1]}.csv

# Load the original CSV file
df = pd.read_csv(f'test/combined_ncu_report_{sys.argv[1]}.csv')

# Delete unwanted columns
columns_to_delete = ['ID','Process ID','Process Name','Host Name','Context']
df = df.drop(columns=columns_to_delete)

# Extract the string from the relevant column (replace 'YourColumnName' with the actual name)
# Extract COO, SortR, and Cities.mtx
df[['Solution', 'Sort', 'Matrix','N_STREAM','BLOCK_SIZE_P','ITER']]  = df['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg'].str.extract(
    r':(\w+)\s+(\w+)\s+([\w\.]+)\s+(\d+)\s+(\d+)\s+(\d+)'
)

df = df.drop(columns=['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg','Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg','Rule Name','Rule Type','Rule Description','Estimated Speedup Type','Estimated Speedup','Device','CC'])

# Extract function name before the first opening parenthesis
df['Kernel Name'] = df['Kernel Name'].str.extract(r'^([^\(]+)')

# Extract the first number inside parentheses
df['Block Size'] = df['Block Size'].str.extract(r'\(\s*(\d+)')

# Extract the first number inside parentheses
df['Grid Size'] = df['Grid Size'].str.extract(r'\(\s*(\d+)')

df = df.rename(columns={
    'Kernel Name': 'Kernel_N',
    'Block Size': 'Block_S',
    'Grid Size': 'Grid_S',
    'Section Name': 'Section_N',
    'Metric Name': 'Metric_N',
    'Metric Unit': 'Metric_U',
    'Metric Value': 'Metric_V',

})



# Save the new CSV
df.to_csv(f'test/combined_ncu_report_new_{sys.argv[1]}.csv', index=False)
