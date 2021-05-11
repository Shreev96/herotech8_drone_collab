import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    # Defining and initializing the architecture of the NN
    def __init__(self, window_size, num_actions):
        super().__init__()
        # Note that the input to the NN is the flattened tensor of the grid state (here, 1 x 25)
        self.fc1 = nn.Linear(window_size ** 2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_actions)

    # Defining an implicit function to carry out the forward pass of the network
    def forward(self, x):
        # TODO: see for using gpu
        x = F.relu(self.fc1(x))  # Applying a relu activation to the 1st FC layer (input)
        x = F.relu(self.fc2(x))  # The same for the hidden layer
        return self.fc3(x)  # Returning the 'Q' values for each of the possible actions
