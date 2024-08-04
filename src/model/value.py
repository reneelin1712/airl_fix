import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
import pandas as pd


class ValueCNN(nn.Module):
    def __init__(self, path_feature, link_feature, input_dim, pad_idx=None, speed_data=None):
        super(ValueCNN, self).__init__()

        # Load speed data
        self.speed_data = speed_data

        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.pad_idx = pad_idx

        self.fc1 = nn.Linear(input_dim, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, 1)  # [batch, 8]

        # Increase the input dimension by 1 to account for the weather feature
        self.fc1 = nn.Linear(input_dim + 1, 120)

    def to_device(self, device):
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)

    def process_features(self, state, des, time_step):
        # print('state', state.shape, 'des', des.shape)
        path_feature = self.path_feature[state, des, :]
        edge_feature = self.link_feature[state, :]

        # Get speed features
        speed_features = []
        for i in range(state.size(0)):
            speed = self.speed_data.get((state[i].item(), time_step[i].item()), 0)  # Default to 0 if not found
            speed_features.append(speed)
        speed_feature = torch.tensor(speed_features, dtype=torch.float32, device=state.device).unsqueeze(-1)


        # # Extract weather feature from the first dimension of state
        # weather_feature = path_feature[:, 0].unsqueeze(-1).float()

        feature = torch.cat([speed_feature, path_feature, edge_feature], -1)
        # feature = torch.cat([path_feature, edge_feature], -1)  # [batch_size, n_path_feature + n_edge_feature]
        return feature

    def forward(self, state, des, time_step):  # 这是policy
        x = self.process_features(state, des, time_step)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x