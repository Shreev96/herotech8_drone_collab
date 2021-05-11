from collections import namedtuple

import numpy as np
import random
from enum import IntEnum

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN


class Agent:
    class Actions(IntEnum):
        """ Possible actions"""
        left = 0
        right = 1
        up = 2
        down = 3

    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state'))

    def __init__(self, i, init_pos, goal, window_size):
        # ID of the agent (represents the integer number to look for on the grid
        self.id = i

        # Position of the agent
        self.pos = init_pos

        # Position of its goal
        self.goal = goal

        # Boolean to know if the agent is done
        self.done = False

        self.actions = Agent.Actions

        # RL stuff

        # Exploration vs Exploitation coefficient for e-greedy algorithm
        self.eps = 0.9

        # Batch size
        self.batch_size = 128

        # Learning rate for the Q-table update equation
        self.learning_rate = 0.1

        # Discount factor for the Q-table update equation
        self.discount_factor = 0.999

        # Neural network
        self.model = DQN(window_size=window_size, num_actions=len(self.actions))

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.max_buffer_size = 100
        self.experience_replay = []

    def select_action(self, observation):
        if random.random() > self.eps:
            with torch.no_grad():
                # reshape the observation (for whatever reason)
                input_tensor = (torch.from_numpy(observation).view(1, -1)).float()  # convert to float because DQN expect float and not double
                # t.argmax(1) will return the index of the maximum value of all elements in the tensor t
                # so it return the action (as an integer) with the larger expected reward.
                return self.model(input_tensor).argmax(1).view(1, 1)
        else:
            return torch.tensor([[random.randrange(len(self.actions))]], dtype=torch.long)

    def add_to_buffer(self, s, a, r, s_):
        """Save a transition in the experience replay memory"""
        s = (torch.from_numpy(s).view(1, -1)).float()  # convert to float because DQN expect float and not double
        s_ = (torch.from_numpy(s_).view(1, -1)).float()  # convert to float because DQN expect float and not double
        self.experience_replay.append(Agent.Transition(s, a, r, s_))

    def sample_buffer(self, batch_size):
        """ Sample a batch of batch_size from the experience replay memory"""
        return random.sample(self.experience_replay, batch_size)

    def train(self):
        """Perform a single step of the optimization"""

        if len(self.experience_replay) < self.batch_size:
            return

        transitions = self.sample_buffer(self.batch_size)

        # Converts batch-array of Transitions to Transition of batch-arrays.
        # [Transition(...), Transition(...)] ---> Transition(observation = [...], ...)
        batch = Agent.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute max_a(Q(s_{t+1}, a)) for all next states (when state is not final)
        # could be done using a target DQN for stability
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        expected_state_action_values = expected_state_action_values.float()
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def __eq__(self, other):
        return self.id == other.id
