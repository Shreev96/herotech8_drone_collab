import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F

from agent import AgentBase


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


class AgentSQN(AgentBase):

    def __init__(self, i, window_size, device, start=None, goal=None):
        super().__init__(i, device, start, goal)

        # SQL stuff
        self.policy_model = SoftQNetwork(window_size=window_size, num_actions=len(self.actions)).to(self.device)
        self.target_model = SoftQNetwork(window_size=window_size, num_actions=len(self.actions)).to(self.device)
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
