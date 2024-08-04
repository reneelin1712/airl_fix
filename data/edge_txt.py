import pandas as pd

# Read the edges.csv file
edges_df = pd.read_csv('edges.csv')

# List of required columns
required_columns = ['u', 'v', 'name', 'highway', 'oneway', 'length', 'lanes', 'bridge', 'ref', 'junction', 'maxspeed', 'tunnel', 'access', 'id', 'cnt', 'n_id']

# Ensure all required columns are present in the DataFrame
for col in required_columns:
    if col not in edges_df.columns:
        edges_df[col] = ''

# Add a column 'cnt' with empty values if it doesn't already exist
if 'cnt' not in edges_df.columns:
    edges_df['cnt'] = ''

# Add a column 'n_id' with the index values
edges_df['n_id'] = edges_df.index

# Reorder the DataFrame to match the required columns order
edges_df = edges_df[required_columns]

# Save the DataFrame to edges.txt with the desired format and headers
edges_df.to_csv('edge.txt', sep=',', index=False, header=True)

print("edges.txt created successfully")
