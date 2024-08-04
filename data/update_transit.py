import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('transit.csv')

# Group the DataFrame by 'link_id' and correct the 'action' column
df['action'] = df.groupby('link_id').cumcount()

# Save the processed DataFrame back to the CSV file
df.to_csv('transit_processed.csv',index=False)