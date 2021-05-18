import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    # Defining and initializing the architecture of the NN
    def __init__(self, device, window_size, num_actions):
        super(DQN, self).__init__()
        self.device = device
        # Note that the input to the NN is the flattened tensor of the grid state (here, 1 x 25)
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(16)

        def conv1d_size_out(size, kernel_size=3, stride=1):
            # Refer to Pytorch website
            return (size - (kernel_size - 1) - 1)//stride + 1

        conv_size = conv1d_size_out(conv1d_size_out(window_size**2)) * 16  # Apply twice since 2x filtering stages
        self.head = nn.Linear(conv_size, num_actions)  # Linear layer at the end

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # TODO: see for using gpu
        x = x.to(self.device)
        x = torch.unsqueeze(x, 1)

        if x.size(1) != 1:
            assert False

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.head(x)  # Returning the 'Q' values for each of the possible actions
