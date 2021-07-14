import configparser

import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F

from agent import AgentBase

from NN import Conv2D_NN


class AgentSQN(AgentBase):

    def __init__(self, i, window_size, device, start=None, goal=None):
        super().__init__(i, device, start, goal)

        # SQL stuff
        self.policy_model = Conv2D_NN(device=self.device, dict_size=len(self.GridLegend),
                                      window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model = Conv2D_NN(device=self.device, dict_size=len(self.GridLegend),
                                      window_size=window_size, num_actions=len(self.actions)).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        config = configparser.ConfigParser()
        config.read("config.ini")
        params = config["SQN Parameters"]

        # Entropy weighting
        self.alpha = params.getfloat("alpha")

        # Batch size
        self.batch_size = params.getint("batch_size")

        # Learning rate for the Q table update equation
        self.learning_rate = params.getfloat("learning_rate")

        # Discount factor for the Q-table update equation
        self.discount_factor = params.getfloat("discount_factor")

        # Step delay before target_model update
        self.update_steps = params.getint("update_period")
        self._learned_steps = 0

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)

    def select_action(self, observation):

        # Shape of observation is (N_batch, Channels_in, Height, Width)
        # observation = torch.FloatTensor(observation).to(self.device)  # No longer need to flatten it
        observation = observation.float()

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
        action_batch = torch.cat(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.cat(batch.reward).unsqueeze(1).to(self.device)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation for this agent ended)
        non_final_mask = ~torch.tensor(batch.done, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat(batch.next_state)[non_final_mask].to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_model(state_batch) \
            .gather(1, action_batch.long())

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

        return loss.item()

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self, path):
        self.policy_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.policy_model.state_dict())
