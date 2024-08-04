import pandas as pd
from yen_ksp import ksp_yen
from context_feature_computation import construct_graph

node_p = "../../data/node.txt"
edge_p = "../../data/edge.txt"
network_p = "../../data/transit.npy"

graph = construct_graph(edge_p, network_p)

# Read the "path.csv" file into a DataFrame
path_df = pd.read_csv("../../data/cross_validation/train_CV0_size1000.csv")

# Extract the "ori" and "des" columns
ori_list = path_df["ori"].tolist()
des_list = path_df["des"].tolist()

# Create an empty list to store the shortest path results
shortest_paths = []

# Iterate over each row of the DataFrame
for ori, des in zip(ori_list, des_list):
    # Find the shortest path using ksp_yen
    candidate_path = ksp_yen(graph, ori, des, 1)
    
    if candidate_path:
        shortest_path = "_".join(map(str, map(int, candidate_path[0]['path']))) 
        path_length = candidate_path[0]['cost']
    else:
        shortest_path = ""
        path_length = 0
    
    # Append the shortest path result to the list
    shortest_paths.append([ori, des, shortest_path, path_length])

# Create a new DataFrame with the shortest path results
shortest_path_df = pd.DataFrame(shortest_paths, columns=["ori", "des", "shortest_path", "path_length"])
shortest_path_df[["ori", "des", "path_length"]] = shortest_path_df[["ori", "des", "path_length"]].astype(int)

# Save the new DataFrame to a CSV file
shortest_path_df.to_csv("../../data/shortest/shortest_paths.csv", index=False)