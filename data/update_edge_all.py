import pandas as pd
import os

# Path to the directory containing the files
folder_path = 'states'

# Get all link_speeds files from the directory, sorted for consistent order
link_speed_files = sorted([f for f in os.listdir(folder_path) if f.startswith('link_speeds')])

# Read the text file containing edge data
edges = pd.read_csv('edge.txt', delimiter=',')

# Initialize an empty DataFrame to store final results
final_results = pd.DataFrame()

# Process each link_speed file
for index, link_speed_file in enumerate(link_speed_files):
    # Read the link speeds CSV file
    link_speed = pd.read_csv(os.path.join(folder_path, link_speed_file))

    # Determine unique time steps based on '100s_interval' column
    unique_intervals = sorted(link_speed['100s_interval'].unique())
    # Create new time_step values that incorporate the file index
    time_steps = pd.DataFrame({'time_step': [10*(index+1) + i for i in range(len(unique_intervals))]})
    
    # Map '100s_interval' to new 'time_step' values
    interval_to_timestep = {interval: time_steps['time_step'][i] for i, interval in enumerate(unique_intervals)}
    link_speed['time_step'] = link_speed['100s_interval'].map(interval_to_timestep)

    # Create a DataFrame for all combinations of n_id from edges and the new time_step values
    edges_extended = pd.merge(edges.assign(key=1), time_steps.assign(key=1), on='key').drop('key', axis=1)

    # Merge the extended edges DataFrame with modified link_speed on n_id and new time_step
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
    original_columns = edges.columns.tolist()
    new_columns = original_columns + ['time_step', 'speed']
    result = result[new_columns]

    # Append results to the final DataFrame
    final_results = pd.concat([final_results, result], ignore_index=True)

# Save to a new TXT file in a similar format to edge.txt
final_results.to_csv('updated_edges.txt', index=False, sep=',', mode='w', header=True)

# Display the head of the DataFrame to check
print(final_results.head())
