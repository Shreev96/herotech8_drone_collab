import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F

from agent import AgentBase


class SQN(nn.Module):
    """Simple DNN"""

    def __init__(self, window_size, num_actions):
        super(SQN, self).__init__()
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


class Conv_SQN(nn.Module):
    # Convolutional version of the SQN

    def __init__(self, window_size, num_actions, device, dict_size):

        super(Conv_SQN, self).__init__()

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

        # x = x.to(self.device)
        # x = torch.unsqueeze(x, 1)  # Add a batch dimension

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.shape[0], -1)  # Equivalent to flattening the tensor

        return self.head(x)  # Returning the 'Q' values for each of the possible actions


class AgentSQN(AgentBase):

    def __init__(self, i, window_size, device, start=None, goal=None):
        super().__init__(i, device, start, goal)

        # SQL stuff
        self.policy_model = Conv_SQN(device=self.device, dict_size=len(self.GridLegend),
                                     window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model = Conv_SQN(device=self.device, dict_size=len(self.GridLegend),
                                     window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())

        # Entropy weighting
        self.alpha = 0.01

        # Batch size
        self.batch_size = 128

        # Learning rate for the Q table update equation
        self.learning_rate = 0.01

        # Discount factor for the Q-table update equation
        self.discount_factor = 0.99

        # Step delay before target_model update
        self.update_steps = 10  # Could also use 2
        self._learned_steps = 0

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)

    def select_action(self, observation):

        # Shape of observation is (N_batch, Channels_in, Height, Width)
        observation = torch.FloatTensor(observation).to(self.device)  # No longer need to flatten it

        with torch.no_grad():
            # Compute soft Action-Value Q values
            q_values = self.policy_model(observation)
            # Compute soft-Value V values
            v_values = self.alpha * torch.logsumexp(q_values / self.alpha, dim=1, keepdim=True)
            # Compute distribution
            dist = torch.exp((q_values - v_values) / self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.view(1,)  # Want a 1D tensor returned

    def train(self):
        if len(self.experience_replay) < self.batch_size:
            return

        self._learned_steps += 1

        if self._learned_steps % self.update_steps == 0:
            # Update the target network, copying all weights and biases
            self.target_model.load_state_dict(self.policy_model.state_dict())

        transitions = self.sample_buffer(self.batch_size)

        # Converts batch-array of Transitions to Transition of batch-arrays.
        # [Transition(...), Transition(...)] ---> Transition(observation = [...], ...)
        batch = AgentBase.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).unsqueeze(1).to(self.device)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation for this agent ended)
        non_final_mask = ~torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(batch.next_state)[non_final_mask].to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_model(state_batch)\
            .gather(1, action_batch.long().reshape(len(action_batch), 1))

        # Compute the expected values
        with torch.no_grad():
            # Compute soft-Q(s_{t+1}, .) for every actions for every states that has a next state
            next_state_q_values = self.target_model(non_final_next_states)
            # Compute soft-V(s_{t+1}) for all the next state
            next_state_values = torch.zeros((self.batch_size, 1), device=self.device)
            next_state_values[non_final_mask] = self.alpha * torch.logsumexp(next_state_q_values / self.alpha,
                                                                             dim=1,
                                                                             keepdim=True)
            # Compute expected Q-values
            expected_q = (next_state_values * self.discount_factor) + reward_batch
            expected_q = expected_q.float()

        loss = F.mse_loss(state_action_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self, path):
        self.policy_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.policy_model.state_dict())

