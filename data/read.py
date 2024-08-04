import numpy as np
import csv
states = np.load('traj_20181030_dX_0900_0930_states.npy', allow_pickle=True)
# actions = np.load('traj_20181030_dX_0900_0930_actions.npy', allow_pickle=True)

print('states',states[0])
# print('actions',actions[0])

# Initialize an empty list to store the formatted data
formatted_data = []

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
        'time_step': timestep/100
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