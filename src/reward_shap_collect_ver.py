from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
import time
import torch
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, \
    minmax_normalization, load_train_sample, load_test_traj
from network_env import RoadWorld
from utils.torch import to_device
import numpy as np
import pandas as pd
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
import csv
import shap
import matplotlib.pyplot as plt

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

cv = 0  # cross validation process [0, 1, 2, 3, 4]
size = 10000  # size of training data [100, 1000, 10000]
gamma = 0.99  # discount factor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv

"""environment"""
edge_p = "../data/edge.txt"
network_p = "../data/transit.npy"
path_feature_p = "../data/feature_od.npy"
train_p = "../data/cross_validation/train_CV%d_size%d.csv" % (cv, size)
test_p = "../data/cross_validation/test_CV%d.csv" % cv
model_p = "../trained_models/airl_CV%d_size%d.pt" % (cv, size)

"""initialize road environment"""
od_list, od_dist = ini_od_dist(train_p)
env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
"""load path-level and link-level feature"""
path_feature, path_max, path_min = load_path_feature(path_feature_p)
edge_feature, link_max, link_min = load_link_feature(edge_p)
path_feature = minmax_normalization(path_feature, path_max, path_min)
path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
edge_feature = minmax_normalization(edge_feature, link_max, link_min)
edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

"""define actor and critic"""
edge_data = pd.read_csv('../data/updated_edges.txt')
speed_data = {(row['n_id'], row['time_step']): row['speed'] for _, row in edge_data.iterrows()}

policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                    path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                    env.pad_idx, speed_data).to(device)
value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                    path_feature_pad.shape[-1] + edge_feature_pad.shape[-1], speed_data=speed_data).to(device)
discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                env.state_action, path_feature_pad, edge_feature_pad,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                env.pad_idx, speed_data).to(device)

# Read the transit data from the CSV file
transit_data = pd.read_csv('../data/transit.csv')

# Create a dictionary to map (link_id, next_link_id) to action
transit_dict = {}
for _, row in transit_data.iterrows():
    transit_dict[(row['link_id'], row['next_link_id'])] = row['action']

def evaluate_rewards(traj_data, time_steps, policy_net, discrim_net, env, transit_dict, transit_data):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)
    discrim_net.to(device)
    
    reward_data = []
    input_features = []
    output_rewards = []
    
    for episode_idx, (traj, time_step) in enumerate(zip(traj_data, time_steps)):
        path = traj.split('_')
        time_step = int(time_step)
        
        des = torch.LongTensor([int(path[-1])]).long().to(device)
        
        for step_idx in range(len(path) - 1):
            state = torch.LongTensor([int(path[step_idx])]).to(device)
            next_state = torch.LongTensor([int(path[step_idx + 1])]).to(device)
            time_step_tensor = torch.LongTensor([time_step]).to(device)
            
            action = transit_dict.get((int(path[step_idx]), int(path[step_idx + 1])), 'N/A')
            action_tensor = torch.LongTensor([action]).to(device) if action != 'N/A' else None
            
            if action_tensor is not None:
                with torch.no_grad():
                    neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action_tensor, next_state)
                    log_prob = policy_net.get_log_prob(state, des, action_tensor, time_step_tensor).squeeze()
                    reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action_tensor, log_prob, next_path_feature, next_edge_feature, time_step_tensor)
                    
                    # Get speed feature
                    speed_feature = policy_net.process_features(state, des, time_step_tensor)
                    
                    # Collect input features
                    input_feature = {
                        'speed_feature': speed_feature.squeeze().cpu().numpy().flatten().tolist(),
                        'neigh_path_feature': neigh_path_feature.cpu().numpy().flatten().tolist(),
                        'neigh_edge_feature': neigh_edge_feature.cpu().numpy().flatten().tolist(),
                        'path_feature': path_feature.cpu().numpy().flatten().tolist(),
                        'edge_feature': edge_feature.cpu().numpy().flatten().tolist(),
                        'next_path_feature': next_path_feature.cpu().numpy().flatten().tolist(),
                        'next_edge_feature': next_edge_feature.cpu().numpy().flatten().tolist(),
                        'action': action_tensor.item(),
                        'log_prob': log_prob.item(),
                        'time_step': time_step
                    }
                    input_features.append(input_feature)
                    output_rewards.append(reward.item())
            else:
                reward = torch.tensor('N/A')
            
            reward_data.append({
                'episode': episode_idx,
                'des': des.item(),
                'step': step_idx,
                'state': path[step_idx],
                'action': action,
                'next_state': path[step_idx + 1],
                'reward': reward.item() if reward != 'N/A' else 'N/A',
                'time_step': time_step,
            })
    
    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)
    
    return reward_df, input_features, output_rewards

def create_shap_explainer(model, input_features):
    # Convert input_features from list of dictionaries to a 2D numpy array
    feature_keys = ['speed_feature', 'neigh_path_feature', 'neigh_edge_feature', 'path_feature', 'edge_feature', 'next_path_feature', 'next_edge_feature', 'action', 'log_prob', 'time_step']
    
    # Initialize an empty list to store flattened features
    flattened_features = []
    
    for feature_dict in input_features:
        # Flatten and concatenate all features for each sample
        sample_features = []
        for key in feature_keys:
            if key in feature_dict:
                if isinstance(feature_dict[key], (int, float)):
                    sample_features.append(feature_dict[key])
                else:
                    sample_features.extend(np.array(feature_dict[key]).flatten())
            else:
                print(f"Warning: {key} not found in feature dictionary")
        flattened_features.append(sample_features)
    
    # Convert to numpy array
    input_features_array = np.array(flattened_features)

    def predict_fn(input_features):
        # Determine the number of samples
        num_samples = input_features.shape[0]

        # Initialize an array to store the model outputs
        model_outputs = np.zeros(num_samples)

        # Iterate over the samples
        for i in range(num_samples):
            # Extract the features for the current sample
            neigh_path_feature = torch.tensor(input_features[i, :117].reshape(9, 13), dtype=torch.float32)  # 9 actions * 13 dimensions = 117
            neigh_edge_feature = torch.tensor(input_features[i, 117:189].reshape(9, 8), dtype=torch.float32)  # 9 actions * 8 dimensions = 72
            speed_feature = torch.tensor(input_features[i, 189:198].reshape(9, 1), dtype=torch.float32)  # 9 actions * 1 dimension = 9
            path_feature = torch.tensor(input_features[i, 198:211], dtype=torch.float32)  # 13 dimensions for path feature
            edge_feature = torch.tensor(input_features[i, 211:219], dtype=torch.float32)  # 8 dimensions for edge feature
            action = torch.tensor(input_features[i, 219], dtype=torch.long)
            log_prob = torch.tensor(input_features[i, 220], dtype=torch.float32)
            next_path_feature = torch.tensor(input_features[i, 221:234], dtype=torch.float32)  # 13 dimensions for next path feature
            next_edge_feature = torch.tensor(input_features[i, 234:242], dtype=torch.float32)  # 8 dimensions for next edge feature
            time_step = torch.tensor(input_features[i, 242], dtype=torch.long)

            # Ensure action is non-negative and within the valid range
            action = torch.clamp(action, min=0, max=model.action_num - 1)

            # Calculate the model output for the current sample
            model_output = model.forward_with_actual_features(
                neigh_path_feature.unsqueeze(0),
                neigh_edge_feature.unsqueeze(0),
                path_feature.unsqueeze(0),
                edge_feature.unsqueeze(0),
                action.unsqueeze(0),
                log_prob.unsqueeze(0),
                next_path_feature.unsqueeze(0),
                next_edge_feature.unsqueeze(0),
                time_step.unsqueeze(0)
            ).detach().numpy()

            # Store the model output for the current sample
            model_outputs[i] = model_output

        return model_outputs

    # Create the background dataset
    background_data = shap.sample(input_features_array, 10) 

    explainer = shap.KernelExplainer(predict_fn, background_data)
    return explainer

def analyze_shap_values(explainer, input_features, feature_indices):
    # Convert input_features from list of dictionaries to a 2D numpy array
    feature_keys = ['speed_feature', 'neigh_path_feature', 'neigh_edge_feature', 'path_feature', 'edge_feature', 'next_path_feature', 'next_edge_feature', 'action', 'log_prob', 'time_step']
    
    # Initialize an empty list to store flattened features
    flattened_features = []
    
    for feature_dict in input_features:
        # Flatten and concatenate all features for each sample
        sample_features = []
        for key in feature_keys:
            if key in feature_dict:
                if isinstance(feature_dict[key], (int, float)):
                    sample_features.append(feature_dict[key])
                else:
                    sample_features.extend(np.array(feature_dict[key]).flatten())
            else:
                print(f"Warning: {key} not found in feature dictionary")
        flattened_features.append(sample_features)
    
    # Convert to numpy array
    input_features_array = np.array(flattened_features)

    # Calculate SHAP values using the explainer
    shap_values = explainer.shap_values(input_features_array)
    shap_values_squeezed = np.squeeze(shap_values)

    # Subset the SHAP values to only analyze selected features
    selected_shap_values = shap_values_squeezed[:, feature_indices]

    # Map selected feature indices to their names
    feature_names_dict = {
    189: 'speed_neighbor_1', 190: 'speed_neighbor_2', 191: 'speed_neighbor_3',
    192: 'speed_neighbor_4', 193: 'speed_neighbor_5', 194: 'speed_neighbor_6',
    195: 'speed_neighbor_7', 196: 'speed_neighbor_8', 197: 'speed_neighbor_9',
    198: 'shortest_distance', 199: 'number_of_links', 200: 'number_of_left_turn',
    201: 'number_of_right_turn', 202: 'number_of_u_turn', 203: 'freq_road_type_1',
    204: 'freq_road_type_2', 205: 'freq_road_type_3', 206: 'freq_road_type_4',
    207: 'freq_road_type_5', 208: 'freq_road_type_6', 209: 'angle',
    210: 'link_length', 211: 'road_type_1', 212: 'road_type_2', 213: 'road_type_3',
    214: 'road_type_4', 215: 'road_type_5', 216: 'road_type_6', 217: 'ratio'
    }
    selected_feature_names = [feature_names_dict.get(index, f'Feature {index}') for index in feature_indices]

    # Convert selected SHAP values to a DataFrame
    selected_shap_values_df = pd.DataFrame(selected_shap_values, columns=selected_feature_names)

    # Save the DataFrame to a CSV file
    selected_shap_values_df.to_csv('selected_shap_values.csv', index=False)
    print("Selected SHAP values saved to 'selected_shap_values.csv'")

    # Plot the SHAP summary plot with selected feature names
    shap.summary_plot(selected_shap_values, input_features_array[:, feature_indices], plot_type="bar", feature_names=selected_feature_names, show=False)
    # Save the plot as an image file
    plt.savefig('selected_shap_summary_plot.png')
    plt.close()
    print("Selected SHAP summary plot saved to 'selected_shap_summary_plot.png'")

# Read the trajectory data from the CSV file
trajectory_data = []
with open('trajectory_with_timestep.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        trajectory_data.append(row)

# Extract test and learner trajectories and their timesteps
test_traj = [row[0] for row in trajectory_data]
test_time_steps = [row[1] for row in trajectory_data]
learner_traj = [row[2] for row in trajectory_data]
learner_time_steps = [row[3] for row in trajectory_data]

# Evaluate rewards
reward_df, input_features, output_rewards = evaluate_rewards(test_traj, test_time_steps, policy_net, discrim_net, env, transit_dict, transit_data)
print('input_features length:', len(input_features))

# Create SHAP explainer
explainer = create_shap_explainer(discrim_net, input_features)

# Example usage:
selected_feature_indices = [
    189, 190, 191, 192, 193, 194, 195, 196, 197,  # Speed features
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,  # Path features
    211, 212, 213, 214, 215, 216, 217  # Edge features
]

# Analyze SHAP values
analyze_shap_values(explainer, input_features[:10], selected_feature_indices)