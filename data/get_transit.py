import pandas as pd
import numpy as np
import ast

# Step 1: Load node data
def load_node_data(file_path):
    df = pd.read_csv(file_path)
    df['neighbors'] = df['neighbors'].apply(ast.literal_eval)
    return df.set_index('osmid')

# Step 2: Parse edge.txt
def load_edge_data(edge_file):
    df = pd.read_csv(edge_file, sep=',')
    return df

# Step 3: Calculate direction between two nodes
def calculate_direction(node_data, node1, node2):
    y1, x1 = node_data.loc[node1, ['y', 'x']]
    y2, x2 = node_data.loc[node2, ['y', 'x']]
    angle = np.arctan2(y2 - y1, x2 - x1)
    return np.degrees(angle) % 360

# Step 4: Assign action based on direction
def assign_action(direction):
    actions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((direction + 22.5) % 360 // 45)
    return index

# Step 5: Generate transit data
def generate_transit_data(node_data, edge_data):
    transit_data = []
    
    for _, edge in edge_data.iterrows():
        u, v = edge['u'], edge['v']
        current_link_id = edge['n_id']
        
        if u in node_data.index and v in node_data.neighbors[u]:
            neighbors = node_data.loc[v, 'neighbors']
            
            for neighbor in neighbors:
                next_edge = edge_data[(edge_data['u'] == v) & (edge_data['v'] == neighbor)]
                if not next_edge.empty:
                    next_link_id = next_edge.iloc[0]['n_id']
                    direction = calculate_direction(node_data, v, neighbor)
                    action = assign_action(direction)
                    
                    transit_data.append({
                        'link_id': current_link_id,
                        'action': action,
                        'next_link_id': next_link_id
                    })
    
    return pd.DataFrame(transit_data)

# Main execution
if __name__ == "__main__":
    edge_file = 'edge.txt'
    node_file = 'nodes_with_neighbors.csv'

    # Load node data
    print("Loading node data...")
    node_data = load_node_data(node_file)

    # Load edge data
    print("Loading edge data...")
    edge_data = load_edge_data(edge_file)

    # Generate transit data
    print("Generating transit data...")
    transit_data = generate_transit_data(node_data, edge_data)

    # Save transit data to CSV
    output_file = 'transit.csv'
    transit_data.to_csv(output_file, index=False)
    print(f"Transit data saved to {output_file}")

    # Print the first 5 rows of transit data
    print("\nFirst 5 rows of transit data:")
    print(transit_data.head())

    print("Processing complete.")