import pandas as pd

# Load the Excel file containing the ratio data
edge_ratio_df = pd.read_excel('edge_frequencies_with_ratio.xlsx')

# Load the edge.txt file into a DataFrame
edge_df = pd.read_csv('edge.txt')

# Merge the edge.txt data with the ratio data based on the 'n_id' (which corresponds to 'Edge' in the Excel file)
merged_df = edge_df.merge(edge_ratio_df, how='left', left_on='n_id', right_on='Edge')

# Drop the 'Edge' and 'Frequency' columns
merged_df = merged_df.drop(columns=['Edge', 'Frequency'])

# Fill NaN values in the 'Ratio' column with 0
merged_df['Ratio'] = merged_df['Ratio'].fillna(0)

# Save the updated data back to a new text file
merged_df.to_csv('edge_with_ratio.txt', index=False)
