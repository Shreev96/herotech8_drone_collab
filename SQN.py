import torch.nn as nn


class SoftQNetwork(nn.Module):
    def __init__(self, window_size, num_actions):
        super(SoftQNetwork, self).__init__()
        # Note that the input to the NN is the flattened tensor of the grid state (here, 1 x window_size ** 2)
        self.fc1 = nn.Linear(window_size ** 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
