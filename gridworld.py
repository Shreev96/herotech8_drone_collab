from enum import IntEnum

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class GridWorld(gym.Env):
    """
    2D Grid world environment
    """

    metadata = {'render.modes': ['human']}

    class LegalActions(IntEnum):
        """ legal actions"""
        left = 0
        right = 1
        up = 2
        down = 3

    class GridLegend(IntEnum):
        FREE = 1
        AGENT = 2
        OBSTACLE = 0
        VISITED = 5

    class UnknownAction(Exception):
        """Raised when an agent try to do an unknown action"""
        pass

    def __init__(self, agents=None, grid=np.ones((5, 5)), partial_obs=False, max_steps=100):
        if agents is None:
            agents = []
        self.agents = agents
        self.grid = grid  # TODO: make it random later

        # Define if the agents use partial observation or global observation
        self.partial_obs = partial_obs

        self.actions = GridWorld.LegalActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=0, high=1, shape=grid.shape, dtype='uint8')

        self.agents_initial_pos = [agent.pos for agent in self.agents]  # starting position of the agents on the grid
        self.agents_visited_cells = {agent: [] for agent in self.agents}  # initialise agents visited cells lists

        self.max_steps = max_steps
        self.step_count = 0

    def _is_legal(self, action, agent):
        """ return True if action is legal"""
        return True

    def reset(self):
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.pos = self.agents_initial_pos[i]
            self.agents_visited_cells[agent] = []
        # TODO: make it random later

        self.step_count = 0
        self.render()  # show the initial arrangement of the grid

        # return first observation
        return self.gen_obs()

    def _reward_agent(self, i):
        """ compute the reward for the i-th agent in the current state"""
        illegal = False
        agent = self.agents[i]
        (n, m) = agent.pos

        # check for out of bounds
        if not (0 <= n < self.grid.shape[0] and 0 <= m < self.grid.shape[1]):
            reward = -2
            agent.done = True
            illegal = True
            # TODO : stop the game on illegal moves?

        # check for collisions with obstacles (statics and dynamics)
        # for now it only checks for obstacles and others agents but it could be generalized with
        # the definition of Cell objects : check if cell is empty or not
        elif (self.grid[n, m] == GridWorld.GridLegend.OBSTACLE  # obstacles
                or (n, m) in [self.agents[j].pos for j in range(len(self.agents)) if j != i]):  # other agents
            reward = -2
            illegal = True
            agent.done = True
            # TODO : stop the game on illegal moves?

        # check if agent reached its goal
        # (does each agent have a specific goal? If yes, is it an attribute of the class Agent?)
        elif (n, m) == agent.goal:
            reward = 10
            agent.done = True

        # penalize for visiting previously visited cells
        elif (n, m) in self.agents_visited_cells[i]:
            reward = -0.5

        # penalise the agent for extra moves (redundant with visited cells in the end, no?)
        else:
            reward = -0.1
            self.agents_visited_cells[agent].append((n, m))

        return reward, illegal

    def step(self, actions):
        self.step_count += 1

        assert len(actions) == len(self.agents), "number of actions must be equal to number of agents"

        # get a random permutation ( agents actions/reward must be order-independent)
        random_order = np.random.permutation(len(self.agents))

        rewards = np.zeros(len(actions))

        for i in random_order:
            assert self._is_legal(actions[i], self.agents[i])

            # agent mission is already done
            if self.agents[i].done:
                continue

            action = actions[i]
            agent = self.agents[i]

            # agent current pos
            (n, m) = agent.pos

            # Go LEFT
            if action == self.actions.left:
                n = -1

            # Go RIGHT
            elif action == self.actions.right:
                n += 1

            # Go UP
            elif action == self.actions.up:
                m += 1

            # Go DOWN
            elif action == self.actions.down:
                m -= 1

            # unknown action
            else:
                raise GridWorld.UnknownAction(" Unknown action")

            # check if obtained move is legal:
            # TODO : ignore or punish illegal moves? if punish : should we stop the game on illegal moves?

            # apply move
            agent.pos = (n, m)

            rewards[i], illegal = self._reward_agent(agent)

        # game over if step_count greater than max_steps or if all the agents are done
        done = self.step_count >= self.max_steps or all(agent.done for agent in self.agents)

        # compute observation
        obs = self.gen_obs()

        return obs, rewards, done, {}

    def gen_obs(self):
        """Generate the observation"""
        return [self._gen_obs_agent(agent) for agent in self.agents]

    def _gen_obs_agent(self, agent):
        """Generate the agent's view"""
        if self.partial_obs:
            # TODO: enable partial observation
            raise Exception("Not implemented yet")
        else:
            canvas = self.grid.copy()

            # mark visited cells
            canvas[tuple(self.agents_visited_cells[agent])] = GridWorld.GridLegend.VISITED

            # mark agents : here there is a risk of contradiction with visited cells
            # canvas[(agent.pos for agent in self.agents)] = GridWorld.GridLegend.AGENT
            for agent in self.agents:
                canvas[agent.pos] = agent.id

            return canvas

    def render(self, mode='human'):
        canvas = self.grid.copy()

        for agent in self.agents:
            # mark the visited cells in 0.6 gray
            canvas[tuple(self.agents_visited_cells[agent])] = 0.6

            # mark the terminal states in 0.9 gray
            canvas[agent.goal] = 0.9

            # mark the current position in 0.3 gray
            canvas[agent.pos] = 0.3

        if mode == "human":
            plt.grid("on")

            ax = plt.gca()
            rows, cols = self.grid.shape
            ax.set_xticks(np.arange(0.5, rows, 1))
            ax.set_yticks(np.arange(0.5, cols, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            plt.imshow(canvas, interpolation='none', cmap='gray')
            return

        else:
            super(GridWorld, self).render(mode=mode)  # just raise an exception for not Implemented mode
