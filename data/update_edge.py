import pandas as pd

# Read the CSV file containing link speeds
link_speed = pd.read_csv('link_speeds_20181030_dX_0900_0930.csv')

# Read the text file containing edge data
edges = pd.read_csv('edge.txt', delimiter=',')

# Create a DataFrame for all combinations of n_id from edges and time_step from 0 to 9
time_steps = pd.DataFrame({'time_step': range(10)})
edges_extended = pd.merge(edges.assign(key=1), time_steps.assign(key=1), on='key').drop('key', axis=1)

# Merge the extended edges DataFrame with link_speed on matching n_id (as link_id) and time_step
result = pd.merge(edges_extended, link_speed, left_on=['n_id', 'time_step'], right_on=['link_id', 'time_step'], how='left')

# Fill missing speeds with 0
result['link_speed'].fillna(0, inplace=True)

# Normalize the speed column using Min-Max scaling
max_speed = result['link_speed'].max()
min_speed = result['link_speed'].min()
result['normalized_speed'] = (result['link_speed'] - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0

# Drop unnecessary columns and rename link_speed to speed
result.drop(columns=['link_id', '100s_interval', 'link_speed'], inplace=True)
result.rename(columns={'normalized_speed': 'speed'}, inplace=True)

# Reorder columns to match the original edge data and include new columns
original_columns = edges.columns.tolist()  # list of original columns from edges DataFrame
new_columns = original_columns + ['time_step', 'speed']  # append the new columns to the list
result = result[new_columns]  # reorder DataFrame to this new column order

# Save to a new TXT file in a similar format to edge.txt
result.to_csv('updated_edges.txt', index=False, sep=',', mode='w', header=True)

# Display the head of the DataFrame to check
print(result.head())
