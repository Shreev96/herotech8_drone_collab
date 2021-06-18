import configparser
import math
import random

import torch
import torch.nn as nn
from torch import optim

from agent import AgentBase

from NN import Conv2D_NN


class AgentDQN(AgentBase):

    def __init__(self, i, window_size, device, start=None, goal=None):
        # ID of the agent (represents the integer number to look for on the grid
        super().__init__(i, device, start, goal)

        # DQN model initialisation
        self.policy_model = Conv2D_NN(device=self.device, dict_size=len(self.GridLegend),
                                      window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model = Conv2D_NN(device=self.device, dict_size=len(self.GridLegend),
                                      window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        # Parse config file and extract DQN model parameters
        config = configparser.ConfigParser()
        config.read("config.ini")
        params = config["DQN Parameters"]

        # Exploration vs Exploitation coefficient for e-greedy algorithm
        self.eps = params.getfloat("eps")

        # Batch size
        self.batch_size = params.getint("batch_size")

        # Learning rate for the Q-table update equation
        self.learning_rate = params.getfloat("learning_rate")

        # Discount factor for the Q-table update equation
        self.discount_factor = params.getfloat("discount_factor")

        # Step delay before target_model update
        self.update_steps = params.getint("update_period")  # Could also use 2
        self._learned_steps = 0  # TODO: Look into this

        # Optimizer and Loss function definition
        self.optimizer = optim.SGD(self.policy_model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, observation):
        if random.random() > self.eps:
            with torch.no_grad():
                # Exploitation
                # input_tensor = (torch.from_numpy(observation).view(1, -1)).float()
                # convert to float because DQN expect float and not double

                observation = observation.float().to(self.device)  # Shape of observation = (N_batch, Channels_in, H, W)

                # t.argmax(1) will return the index of the maximum value of all elements in the tensor t
                # so it return the action (as an integer) with the larger expected reward.
                return self.policy_model(observation).max(1)[1].view(1,)  # Want a 1D tensor returned
        else:
            return torch.tensor([random.randrange(len(self.actions))],
                                device=self.device, dtype=torch.long)  # TODO: Check if this is a 1D tensor

    def train(self):
        """Perform a single step of the optimization"""

        if len(self.experience_replay) < self.batch_size:
            return

        self._learned_steps += 1

        if self._learned_steps % self.update_steps == 0:
            # Update the target network, copying all weights and biases
            self.target_model.load_state_dict(self.policy_model.state_dict())

        # # Exponentially decay epsilon
        # self.eps = 0.05 + (0.9 - 0.05) * math.exp(-1 * self._learned_steps / 1000)  # TODO: Need to adjust denominator

        # Linearly decay epsilon
        self.eps = 0.05 + (0.9 - 0.05) * (self._learned_steps / 1000)  # TODO: Need to adjust denominator

        transitions = self.sample_buffer(self.batch_size)

        # Converts batch-array of Transitions to Transition of batch-arrays.
        # [Transition(...), Transition(...)] ---> Transition(observation = [...], ...)
        batch = AgentBase.Transition(*zip(*transitions))

        # Group the states, actions and rewards accumulated during each timestep in the batch
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.cat(batch.reward).unsqueeze(1).to(self.device)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation for this agent ended)
        non_final_mask = ~torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(batch.next_state)[non_final_mask].to(self.device)

        # Compute Q(s_t, a)
        q_values = self.policy_model(state_batch).gather(1, action_batch.long())

        # Compute the expected values
        with torch.no_grad():
            # Compute max_a(Q(s_{t+1}, a)) for all next states (when state is not final)
            # could be done using a target DQN for stability
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]

            # Compute the expected Q values
            expected_q_values = (next_state_values * self.discount_factor) + torch.squeeze(reward_batch)
            expected_q_values = expected_q_values.float()

        # Compute Huber loss
        loss = self.criterion(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)  # 'clamp' Ensures all gradients are between (-1, 1)

        self.optimizer.step()

    def __eq__(self, other):
        return self.id == other.id

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self, path):
        self.policy_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.policy_model.state_dict())
