import pandas as pd

# Read the nodes.csv file
nodes_df = pd.read_csv('nodes.csv')

# Extract the necessary columns and rename them
nodes_df = nodes_df[['osmid', 'y', 'x']]

# Save the DataFrame to nodes.txt with the desired format
nodes_df.to_csv('node.txt', sep=' ', index=False, header=True)

print("nodes.txt created successfully")
