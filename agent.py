import random
from collections import namedtuple, deque
from enum import IntEnum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class AgentBase:
    """
    Base class for Agent object
    """

    class Actions(IntEnum):
        """ Possible actions"""
        left = 0
        right = 1
        up = 2
        down = 3

    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', 'done'))

    def __init__(self, i, device, start=None, goal=None):
        # ID of the agent (represents the integer number to look for on the grid
        self.id = i

        # Position of the agent
        self.init_pos = start
        self.pos = start

        # Position of its goal
        self.init_goal = goal
        self.goal = goal

        # Boolean to know if the agent is done
        self.done = False

        self.actions = AgentBase.Actions

        # DRL stuff
        self.max_buffer_size = 100
        self.experience_replay = deque([], maxlen=10000)

        # PyTorch stuff
        self.device = device

    def add_to_buffer(self, s, a, r, s_, done):
        """Save a transition in the experience replay memory"""
        s = (torch.from_numpy(s).view(1, -1)).float()  # convert to float because DQN expect float and not double
        s_ = (torch.from_numpy(s_).view(1, -1)).float()  # convert to float because DQN expect float and not double
        self.experience_replay.append(AgentBase.Transition(s, a, r, s_, done))

    def sample_buffer(self, batch_size):
        """ Sample a batch of batch_size from the experience replay memory"""
        return random.sample(self.experience_replay, batch_size)

    def select_action(self, observation):
        """
        select an action from the Action set based on an observation of the environment
        :param observation: np.ndarray : observation of the environment
        :return: an action of Agent.Actions
        """
        raise NotImplementedError("Must implement this method")

    def train(self):
        """Perform a single step of the optimization"""
        raise NotImplementedError("Must implement this method")
