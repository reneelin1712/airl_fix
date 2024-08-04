# import pandas as pd

# # Read the "path.csv" file into a DataFrame
# path_df = pd.read_csv("../../data/cross_validation/train_CV0_size1000.csv")

# # Read the "shortest_paths.csv" file into a DataFrame
# shortest_path_df = pd.read_csv("../../data/shortest/shortest_paths.csv")

# # Concatenate the two DataFrames side by side
# merged_df = pd.concat([path_df, shortest_path_df["shortest_path"]], axis=1)

# # Compare the "path" and "shortest_path" columns
# merged_df["Same"] = merged_df.apply(lambda row: "Yes" if row["path"] == row["shortest_path"] else "No", axis=1)

# # Select the desired columns for the output
# output_df = merged_df[["ori", "des", "path", "shortest_path", "Same"]]

# # Save the comparison result to a new CSV file
# output_df.to_csv("../../data/shortest/comparison_result.csv", index=False)


import pandas as pd
from yen_ksp import ksp_yen
from context_feature_computation import construct_graph

# Load the data and construct the graph
node_p = "../../data/node.txt"
edge_p = "../../data/edge.txt"
network_p = "../../data/transit.npy"

graph = construct_graph(edge_p, network_p)

# Load the original training data
original_df = pd.read_csv("../../data/cross_validation/train_CV0_size1000.csv")

# Function to check if a path is the shortest
def is_shortest_path(graph, ori, des, path, length):
    candidate_path = ksp_yen(graph, ori, des, 1)
    if candidate_path:
        shortest_path = "_".join(map(str, map(int, candidate_path[0]['path'])))
        shortest_length = candidate_path[0]['cost']
        return path == shortest_path or length == shortest_length
    return False

# Compare paths
is_shortest = []
for _, row in original_df.iterrows():
    is_shortest.append(is_shortest_path(graph, row['ori'], row['des'], row['path'], row['len']))

# Count how many original paths are the shortest
shortest_count = sum(is_shortest)

# Calculate the percentage
total_paths = len(original_df)
shortest_percentage = (shortest_count / total_paths) * 100

print(f"Number of original paths that are the shortest: {shortest_count} out of {total_paths}")
print(f"Percentage of original paths that are the shortest: {shortest_percentage:.2f}%")

# Save the comparison results
original_df['is_shortest'] = is_shortest
original_df.to_csv("../../data/path_comparison.csv", index=False)