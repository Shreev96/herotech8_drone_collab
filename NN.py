import torch
import torch.nn as nn
import torch.nn.functional as F

# Repository of the NN models created so far


class Conv2D_NN(nn.Module):
    """2D Convolutional version applicable for a DQN & SQN"""

    # Input: Gridworld converted to a list of tensors acc. to the dictionary size
    # Output: Tensor of Q values for each possible action

    def __init__(self, window_size, num_actions, device, dict_size):

        super(Conv2D_NN, self).__init__()

        self.window_size = window_size  # (square root) Size of the grid that the network sees
        self.device = device

        self.kernel_size = (3, 3)  # Square kernel
        self.stride = (1, 1)  # Equal and continuous stride size due to small grid

        self.conv1 = nn.Conv2d(dict_size, dict_size*2, kernel_size=self.kernel_size, stride=self.stride)
        self.conv2 = nn.Conv2d(dict_size*2, 32, kernel_size=self.kernel_size, stride=self.stride)

        def conv2d_size_out(shape, kernel_size=(3, 3), stride=(1, 1)):
            """Size of array after convolution(s). Refer to Pytorch website"""

            h = (shape[0] - (kernel_size[0] - 1) - 1)//stride[0] + 1  # height of convolved array
            w = (shape[1] - (kernel_size[1] - 1) - 1)//stride[1] + 1  # width of convolved array

            shape_out = (h, w)

            return shape_out

        # Apply twice since 2x filtering stages
        conv_size = conv2d_size_out(conv2d_size_out((window_size, window_size), self.kernel_size, self.stride),
                                    self.kernel_size, self.stride)

        linear_input_size = conv_size[0] * conv_size[1] * 32  # 32 is the filter dimension

        # Linear layer to convert the convolved grid to actions
        self.head = nn.Linear(linear_input_size, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)  # Equivalent to flattening the tensor

        return self.head(x)  # Returning the 'Q' values for each of the possible actions


class Conv1D_NN(nn.Module):
    """1D Convolutional version applicable for a DQN & SQN"""

    # Input: 1 x 25 or 1 x 50 or 1 x 75... flattened grid depending on the number of features to be explained to NN
    # # with positions of the obstacles on the first set, and free spaces on the next, goal states on the next etc.
    # Output: Tensor of Q values for each possible action

    # Defining and initializing the architecture of the NN
    def __init__(self, device, window_size, num_actions):

        super(Conv1D_NN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv1d(1, 8, kernel_size=(3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm1d(16)

        def conv1d_size_out(size, kernel_size=3, stride=1):
            # Refer to Pytorch website
            return (size - (kernel_size - 1) - 1)//stride + 1

        conv_size = conv1d_size_out(conv1d_size_out(window_size**2)) * 16  # Apply twice since 2x filtering stages
        self.head = nn.Linear(conv_size, num_actions)  # Linear layer at the end

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = x.to(self.device)
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)

        return self.head(x)


class Linear_NN(nn.Module):
    """Vanilla linear NN applicable for the older versions of the DQN and SQN (without grid2tensor implemented)"""

    # Input: 1 x 25 flattened grid (positions of the free positions and the obstacles normally works)
    # Output: Tensor of Q values for each possible action

    def __init__(self, device, window_size, num_actions):
        super().__init__()
        self.device = device
        # Note that the input to the NN is the flattened tensor of the grid state (here, 1 x window_size ** 2)
        self.fc1 = nn.Linear(window_size ** 2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, num_actions)

    # Defining an implicit function to carry out the forward pass of the network
    def forward(self, x):

        x = x.to(self.device)
        x = F.relu(self.fc1(x))  # Applying a relu activation to the 1st FC layer (input)
        x = F.relu(self.fc2(x))  # The same for the hidden layer

        return self.fc3(x)
