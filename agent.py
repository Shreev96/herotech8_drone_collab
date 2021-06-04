import random
from collections import namedtuple, deque
from enum import IntEnum

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from SQN import SoftQNetwork
from DQN_Conv import DQN


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

    def __init__(self, i, init_pos, goal, device):
        # ID of the agent (represents the integer number to look for on the grid
        self.id = i

        # Position of the agent
        self.pos = init_pos

        # Position of its goal
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


class AgentSQN(AgentBase):

    def __init__(self, i, init_pos, goal, window_size, device):
        super().__init__(i, init_pos, goal, device)

        # SQL stuff
        self.policy_model = SoftQNetwork(window_size=window_size, num_actions=len(self.actions)).to(self.device)
        self.target_model = SoftQNetwork(window_size=window_size, num_actions=len(self.actions)).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        # Entropy weighting
        self.alpha = 0.01

        # Batch size
        self.batch_size = 128

        # Learning rate for the Q-table update equation
        self.learning_rate = 0.01

        # Discount factor for the Q-table update equation
        self.discount_factor = 0.99

        # Step delay before target_model update
        self.update_steps = 10  # Could also use '2'
        self._learned_steps = 0

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)

    def select_action(self, observation):
        observation = torch.FloatTensor(observation).view(1, -1).to(self.device)
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
        return a.view(1, 1)

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
        state_action_values = self.policy_model(state_batch).gather(1, action_batch.long())

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


class AgentDQN(AgentBase):

    def __init__(self, i, init_pos, goal, window_size, device):
        # ID of the agent (represents the integer number to look for on the grid
        super().__init__(i, init_pos, goal, device)

        # DQN stuff

        # Exploration vs Exploitation coefficient for e-greedy algorithm
        self.eps = 0.9

        # Batch size
        self.batch_size = 128

        # Learning rate for the Q-table update equation
        self.learning_rate = 0.1

        # Discount factor for the Q-table update equation
        self.discount_factor = 0.9

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
