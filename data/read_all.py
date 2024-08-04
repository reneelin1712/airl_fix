import numpy as np
import csv
import os

# Path to the directory containing the .npy files
folder_path = 'states'

# List to hold all the filenames that start with 'traj_'
files = [f for f in os.listdir(folder_path) if f.startswith('states_traj') and f.endswith('.npy')]

# Sort files for consistent processing order
files.sort()

print('files',files)
# Initialize an empty list to store the formatted data
formatted_data = []

# Process each file in the folder
for index, file in enumerate(files):
    # Load the states from the current file
    states = np.load(os.path.join(folder_path, file), allow_pickle=True)

    for trajectory in states:
        # Extract the first column of each step in the trajectory
        path = trajectory[:, 0].astype(int)

        # Get the origin and destination nodes
        ori = path[0]
        des = path[-1]

        # Convert the path to a string format
        path_str = "_".join(map(str, path))

        # Get the length of the path
        length = len(path)

        # Get the speed value from the second column of the first step
        timestep = trajectory[0, 3]

        # Create a dictionary with the formatted data
        formatted_row = {
            'ori': ori,
            'des': des,
            'path': f"{path_str}",
            'len': length,
            'time_step': int(f"{index+1}{int(timestep/100)}")  # Modified time_step with file index
        }

        # Append the formatted row to the list
        formatted_data.append(formatted_row)

# Specify the output file path
output_file = "formatted_data_w_timestep.csv"

# Write the formatted data to a CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['ori', 'des', 'path', 'len', 'time_step'])
    writer.writeheader()
    writer.writerows(formatted_data)

print(f"Formatted data saved to: {output_file}")
