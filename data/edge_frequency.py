import pandas as pd
from collections import Counter

# Load the CSV file
df = pd.read_csv('./cross_validation/train_CV0_size10000.csv')

# Split the 'path' column into individual edges
edges = []
for path in df['path']:
    edges.extend(path.split('_'))

# Calculate the frequency of each edge
edge_freq = Counter(edges)

# Convert the frequency dictionary to a DataFrame
edge_freq_df = pd.DataFrame(edge_freq.items(), columns=['Edge', 'Frequency'])

# Calculate the total number of edge occurrences in the dataset
total_edges = sum(edge_freq_df['Frequency'])

# Calculate the ratio for each edge
edge_freq_df['Ratio'] = edge_freq_df['Frequency'] / total_edges

# Ensure the Ratio column is a numeric type
edge_freq_df['Ratio'] = edge_freq_df['Ratio'].astype(float)
edge_freq_df['Edge'] = edge_freq_df['Edge'].astype(int)

# Save the result to an Excel file
edge_freq_df.to_excel('edge_frequencies_with_ratio.xlsx', index=False)