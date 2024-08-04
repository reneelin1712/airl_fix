import numpy as np
import csv
import pandas as pd

def find_and_delete_link_id_pairs(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Create a dictionary to store the mapping of link_id to next_link_id
    link_to_next = {}

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        link_id = row['link_id']
        next_link_id = row['next_link_id']

        # Add the mapping of link_id to next_link_id
        if link_id not in link_to_next:
            link_to_next[link_id] = set()
        link_to_next[link_id].add(next_link_id)

    # Find the pairs where link_id and next_link_id can take actions to each other
    link_id_pairs = []
    for link_id, next_link_ids in link_to_next.items():
        for next_link_id in next_link_ids:
            if next_link_id in link_to_next and link_id in link_to_next[next_link_id]:
                link_id_pairs.append((link_id, next_link_id))

    # Save the link ID pairs to a file
    with open('link_id_pairs.txt', 'w') as file:
        for pair in link_id_pairs:
            file.write(f"({pair[0]}, {pair[1]})\n")

    # Delete the rows corresponding to the link ID pairs
    for pair in link_id_pairs:
        link_id, next_link_id = pair
        df = df.drop(df[(df['link_id'] == link_id) & (df['next_link_id'] == next_link_id)].index)
        df = df.drop(df[(df['link_id'] == next_link_id) & (df['next_link_id'] == link_id)].index)

    return df

# Usage example
csv_file_path = 'transit.csv'
updated_df = find_and_delete_link_id_pairs(csv_file_path)

# Save the updated DataFrame to a new CSV file
updated_df.to_csv('updated_transit.csv', index=False)
