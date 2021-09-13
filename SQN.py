import configparser

import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F

from agent import AgentBase

from NN import Conv2D_NN


class AgentSQN(AgentBase):

    def __init__(self, i, obs_shape, device, start=None, goal=None, config=None):
        super().__init__(i, device, start, goal)

        # SQL stuff
        self.policy_model = Conv2D_NN(device=self.device, dict_size=obs_shape[1],
                                      window_size=obs_shape[-1], num_actions=len(self.actions)).to(self.device)

        self.target_model = Conv2D_NN(device=self.device, dict_size=obs_shape[1],
                                      window_size=obs_shape[-1], num_actions=len(self.actions)).to(self.device)

        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        if config is None:
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

    def select_greedy_action(self, observation):
        observation = observation.float()
        return self.policy_model(observation).max(1)[1].view(1,)


    def train(self):
        if len(self.experience_replay) < self.batch_size:
            return 0

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

        self._learned_steps += 1

        if self._learned_steps % self.update_steps == 0:
            # Update the target network, copying all weights and biases
            self.target_model.load_state_dict(self.policy_model.state_dict())
            self._learned_steps = 0

        return loss.item()

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self, path):
        self.policy_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.policy_model.state_dict())


class CoordinatorSQN(AgentSQN):
    def __init__(self, agents, obs_shape, device, config=None):
        super().__init__(0, obs_shape, device, start=None, goal=None, config=config)
        self.agents = agents

    def train(self) -> float:
        loss = super().train()

        # if self._learned_steps % self.update_steps == 0:
        for agent in self.agents:
            agent.policy_model.load_state_dict(self.policy_model.state_dict())

        return loss
