import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from agent import AgentBase


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


class AgentDQN(AgentBase):

    def __init__(self, i, window_size, device, start=None, goal=None):
        # ID of the agent (represents the integer number to look for on the grid
        super().__init__(i, device, start, goal)

        # DQN stuff

        # Exploration vs Exploitation coefficient for e-greedy algorithm
        self.eps = 0.9

        # Batch size
        self.batch_size = 128

        # Learning rate for the Q-table update equation
        self.learning_rate = 0.1

        # Discount factor for the Q-table update equation
        self.discount_factor = 0.99

        # Step delay before target_model update
        self.update_steps = 10  # Could also use 2
        self._learned_steps = 0

        # Neural network
        self.policy_model = DQN(self.device, window_size=window_size, num_actions=len(self.actions)).to(self.device)
        self.target_model = DQN(self.device, window_size=window_size, num_actions=len(self.actions)).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.SGD(self.policy_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, observation):
        if random.random() > self.eps:
            with torch.no_grad():
                # reshape the observation (for whatever reason)
                input_tensor = (torch.from_numpy(observation).view(1,
                                                                   -1)).float()  # convert to float because DQN expect float and not double
                # t.argmax(1) will return the index of the maximum value of all elements in the tensor t
                # so it return the action (as an integer) with the larger expected reward.
                return self.policy_model(input_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.actions))]],
                                device=self.device, dtype=torch.long)

    def train(self):
        """Perform a single step of the optimization"""

        if len(self.experience_replay) < self.batch_size:
            return

        self._learned_steps += 1

        self.eps = 0.05 + (0.9 - 0.05) * math.exp(-1 * self._learned_steps / 1000)

        if self._learned_steps % self.update_steps == 0:
            # Update the target network, copying all weights and biases
            self.target_model.load_state_dict(self.policy_model.state_dict())

        transitions = self.sample_buffer(self.batch_size)

        # Converts batch-array of Transitions to Transition of batch-arrays.
        # [Transition(...), Transition(...)] ---> Transition(observation = [...], ...)
        batch = AgentBase.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation for this agent ended)
        non_final_mask = ~torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(batch.next_state)[non_final_mask].to(self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Compute max_a(Q(s_{t+1}, a)) for all next states (when state is not final)
        # could be done using a target DQN for stability
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        expected_state_action_values = expected_state_action_values.float()

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __eq__(self, other):
        return self.id == other.id
