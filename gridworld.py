import random
from enum import IntEnum
from typing import Tuple, Dict, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

from rendering import fill_coords, point_in_rect, highlight_img, downsample

from copy import deepcopy
import torch


class WorldObj:  # not used yet
    """
    Base class for grid world objects
    """

    def __init__(self):
        self.pos = None

    def encode(self) -> Tuple[int, ...]:
        """Encode the description of this object"""
        raise NotImplementedError

    def on_entering(self, agent) -> ():
        """Action to perform when an agent enter this object"""
        raise NotImplementedError

    def on_leaving(self, agent) -> ():
        """Action to perform when an agent exit this object"""
        raise NotImplementedError

    def can_overlap(self) -> bool:
        """Can an agent overlap this object?"""
        return True

    def render(self, r) -> ():
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Grid:  # not used yet
    """
    Base class for grids and operations on it (not used yet)
    """
    # Type hints
    _obj_2_idx: Dict[Optional[WorldObj], int]

    # Static cache of pre-rendered tiles
    tile_cache = {}

    class EncodingError(Exception):
        """Exception raised for missing entry in _obj_2_idx"""
        pass

    def __init__(self, width: int, height: int):
        """Create an empty Grid"""
        self.width = width
        self.height = height

        self.grid = np.empty(shape=(width, height), dtype=WorldObj)

        self._obj_2_idx = {None: 0}
        self._idx_2_obj = {v: k for k, v in self._obj_2_idx.items()}

    @classmethod
    def from_array(cls, array: np.ndarray):
        (width, height) = array.shape
        out = cls(width, height)
        out.grid = array
        return out

    @property
    def obj_2_idx(self):
        return self._obj_2_idx

    def __contains__(self, item):
        return item in self.grid

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid1, grid2)

    def __getitem__(self, item):
        out = self.grid.__getitem__(item)
        if isinstance(out, WorldObj):
            return out
        else:
            # slice
            return Grid.from_array(out)

    def __setitem__(self, key, value):
        if isinstance(value, Grid):
            self.grid.__setitem__(key, value.grid)
        else:
            self.grid.__setitem__(key, value)

    def set(self, i, j, v):
        """Set an element of the grid"""
        assert 0 <= i < self.width, "i index out of bounds"
        assert 0 <= j < self.height, "j index out of bounds"
        self.grid[i, j] = v

    def get(self, i, j):
        """Get an element of the grid"""
        assert 0 <= i < self.width, "i index out of bounds"
        assert 0 <= j < self.height, "j index out of bounds"
        return self.grid[i, j]

    def slice(self, top_x, top_y, width, height):
        """Get a subset of the grid"""
        assert 0 <= top_x < self.width
        assert 0 <= top_x + width < self.width
        assert 0 <= top_y + width < self.height
        assert 0 <= top_y < self.height

        return Grid.from_array(self.grid[top_x:(top_x + width), top_y:(top_y + height)])

    @classmethod
    def render_tile(cls, obj: WorldObj, highlight=False, tile_size=32, subdivs=3):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Down-sample the image to perform super-sampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size=32, highlight_mask=None):
        """
        Render this grid at a given scale
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                tile_img = Grid.render_tile(
                    cell,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask: np.ndarray = None):
        """
        Produce a compact numpy encoding of the grid with tuples for each cells
        :param vis_mask: numpy array of boolean as a vision mask
        :return: numpy array
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        assert vis_mask.shape == self.grid.shape

        array = np.zeros((self.width, self.height, 2), dtype="uint8")  # TODO: enable variable length encoding?

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        try:
                            array[i, j, 0] = self._obj_2_idx[None], 0
                        except KeyError:
                            raise Grid.EncodingError("Empty grid cell encoding index not specified")
                    if v is not None:
                        try:
                            array[i, j, 0] = self._obj_2_idx[v.__class__]
                        except KeyError:
                            raise Grid.EncodingError(f"Grid cell encoding index for {v.__class__} not specified")
                        array[i, j, :] = v.encode()

        return array

    @classmethod
    def decode(cls, array):
        """
        Decode an array grid encoding back into a grid using this grid encoding
        :param array: an array grid encoded
        :return: grid
        """

        width, height, channels = array.shape
        assert channels == 2  # TODO: enable variable length encoding?

        grid = cls(width, height)

        for i in range(width):
            for j in range(height):
                type_idx, arg = array[i, j]
                # TODO : continue


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
        # VISITED = 5  # Commented out for now since it interferes with the conv-net input tensor
        # OUT_OF_BOUNDS = 6  # Commented out for now since it interferes with the conv-net input tensor
        GOAL = 7

    class UnknownAction(Exception):
        """Raised when an agent try to do an unknown action"""
        pass

    def __init__(self, agents=None, grid=np.ones((5, 5)), partial_obs=False, width=5, height=5,
                 col_wind=None, range_random_wind=0, probabilities=None):

        if agents is None:
            agents = []
        self.agents = agents
        self.grid = grid  # TODO: make it random later -- DONE

        if probabilities is None and range_random_wind == 0:
            probabilities = [1]  # Zero noise

        # Define if the agents use partial observation or global observation
        self.partial_obs = partial_obs
        if self.partial_obs:
            self.agent_view_width = width
            self.agent_view_height = height

        self.actions = GridWorld.LegalActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=0, high=1, shape=grid.shape, dtype='uint8')

        self.agents_initial_pos = [agent.pos for agent in self.agents]  # starting position of the agents on the grid

        # Wind effects -- TODO: May need to be moved into the world object? or is it okay here?
        self.np_random, _ = self.seed()  # Seeding the random number generator

        if col_wind is None:
            col_wind = np.zeros((len(self.grid, )))

        self.col_wind = col_wind  # Static wind  (rightwards is positive)

        self.range_random_wind = range_random_wind  # Random (dynamic) wind added on top of static wind
        self.w_range = np.arange(-self.range_random_wind, self.range_random_wind + 1)
        self.probabilities = probabilities  # Stochasticity implemented through noise
        assert sum(self.probabilities) == 1

    def reset(self):
        # gen new grid
        # TODO

        # w, h = self.grid.shape
        # free_cells = [(i, j) for i in range(w) for j in range(h) if self.grid[i, j] == self.GridLegend.FREE]

        self.agents_initial_pos = [agent.init_pos for agent in self.agents]

        # place agents
        for i in range(len(self.agents)):
            agent = self.agents[i]
            # reset initial positions
            if agent.init_pos is None:
                # agent does not have a specified unique initial position
                # assign random one
                w, h = self.grid.shape
                start = random.randrange(0, w), random.randrange(0, h)
                while start in self.agents_initial_pos or self.grid[start] == self.GridLegend.OBSTACLE:
                    # while the new start position is the same of another agents
                    # or if the new start position is an obstacle
                    # try another one
                    # (should work fine for not too complex labyrinth or when there is not much agents and be faster
                    # than going through the grid and removing the unavailable cells in such a case)
                    start = random.randrange(0, w), random.randrange(0, h)
                agent.pos = start
                self.agents_initial_pos[i] = start
            else:
                agent.pos = agent.init_pos
                self.agents_initial_pos[i] = agent.init_pos

            # reset goals
            if agent.init_goal is None:
                # agent does not have a specified unique goal
                # assign a random one
                w, h = self.grid.shape
                goal = random.randrange(0, w), random.randrange(0, h)
                while goal == agent.pos or self.grid[goal] == self.GridLegend.OBSTACLE:
                    # while the new goal position is an obstacle or the starting position of the agent
                    # try another one
                    # (should work fine for not too complex labyrinth or when there is not much agents and be faster
                    # than going through the grid and removing the unavailable cells in such a case)
                    goal = random.randrange(0, w), random.randrange(0, h)
                agent.goal = goal
            else:
                agent.goal = agent.init_goal

            agent.done = False

        # self.render()  # show the initial arrangement of the grid

        # return first observation
        return self.gen_obs()

    def trans_function(self, state, action, noise):
        """Creating transition function based on environmental factors
        For now, only wind considered -> static + random (pre-defined probabilities that the agent can
        figure out through experience)"""

        n, m = state

        if self.col_wind[n] != 0:
            wind = self.col_wind[n] + noise

        else:
            wind = 0  # Irrespective of random noise

        # Go UP
        if action == self.actions.up:
            (n, m) = (n - 1, m + wind)

        # Go DOWN
        elif action == self.actions.down:
            (n, m) = (n + 1, m + wind)

        # Go LEFT
        elif action == self.actions.left:
            (n, m) = (n, m - 1 + wind)

        # Go RIGHT
        elif action == self.actions.right:
            (n, m) = (n, m + 1 + wind)

        return n, m

    def _reward_agent(self, i, move):
        """ compute the reward for the i-th agent in the current state"""
        illegal = False
        agent = self.agents[i]
        (n, m) = move

        # check for out of bounds
        if not (0 <= n < self.grid.shape[0] and 0 <= m < self.grid.shape[1]):
            reward = -0.8
            # agent.done = True
            illegal = True

        # check for collisions with obstacles (statics and dynamics)
        # for now it only checks for obstacles and others agents but it could be generalized with
        # the definition of Cell objects : check if cell is empty or not
        elif (self.grid[n, m] == self.GridLegend.OBSTACLE  # obstacles
              or (n, m) in [self.agents[j].pos for j in range(len(self.agents)) if j != i]):  # other agents
            reward = -0.75
            illegal = True
            # agent.done = True

        # check if agent reached its goal
        # (does each agent have a specific goal? If yes, is it an attribute of the class Agent?)
        elif (n, m) == agent.goal:
            reward = 1.0
            agent.done = True

        # # penalize for visiting previously visited cells
        # elif (n, m) in self.agents_visited_cells[i]:
        #     reward = -0.5

        # penalise the agent for extra moves
        else:
            reward = -0.04
            # self.agents_visited_cells[agent].append((n, m))

        return reward, illegal

    def step(self, actions):

        assert len(actions) == len(self.agents), "number of actions must be equal to number of agents"

        # get a random permutation ( agents actions/reward must be order-independent)
        # random_order = np.random.permutation(len(self.agents))

        # keep a backup of agents state
        moves = [(None, None) for _ in self.agents]

        rewards = np.zeros(len(actions))

        # compute the moves
        for i in range(len(self.agents)):

            # agent mission is already done
            if self.agents[i].done:
                continue

            action = actions[i]
            agent = self.agents[i]

            # agent current pos
            (n, m) = agent.pos  # n is the row number, m is the column number

            # Adding random noise (wind) to the action
            noise = self.np_random.choice(self.w_range, 1, p=self.probabilities)[0]

            # Generate move
            (n_, m_) = self.trans_function((n, m), action, noise)

            # Store backup of move for each agent (i)
            moves[i] = (n_, m_)

        # compute rewards and apply moves if they are legal
        # TODO: improve that to take crashes between agents in account
        for i in range(len(self.agents)):
            # compute rewards and illegal assertions
            rewards[i], illegal = self._reward_agent(i, moves[i])

            # apply move if legal
            if not illegal:
                self.agents[i].pos = moves[i]

        # game over if all the agents are done
        done = all(agent.done for agent in self.agents)

        # compute observation
        obs = self.gen_obs()

        return obs, rewards, done, {}

    def gen_obs(self, tensor=1):
        """Generate the observation"""
        return [self._gen_obs_agent(agent, tensor) for agent in self.agents]

    def _gen_obs_agent(self, agent, tensor=1):
        """Generate the agent's view"""
        if self.partial_obs:
            x, y = agent.pos
            w, h = self.agent_view_width, self.agent_view_height
            sub_grid = np.full((w, h), self.GridLegend.OUT_OF_BOUNDS)

            # compute sub_grid corners
            top_x, top_y = x - w // 2, y - h // 2

            for j in range(0, h):
                for i in range(0, w):
                    n = top_x + i
                    m = top_y + j

                    if 0 <= n < self.grid.shape[0] and 0 <= m < self.grid.shape[1]:
                        sub_grid[i, j] = self.grid[n, m]

            return sub_grid

        else:
            canvas = self.grid.copy()

            for a in self.agents:
                canvas[a.pos] = self.GridLegend.AGENT

            # only mark the goal of the agent (not the ones of the others)
            canvas[agent.goal] = self.GridLegend.GOAL

            # Convert to len(dict)-dimensional tensor for Conv_SQN. Can turn on or off
            if tensor == 1:
                canvas = self.grid2tensor(canvas, agent)

            return canvas

    def eval_value_func(self, agent_id):
        """Function to evaluate the value of each position in the grid,
        given a unique agent ID (context for values)"""

        # Simulate observations from each free space in the grid
        # 'obs' is typically a 5 x 5 numpy array (for now)
        free_spaces = np.where(self.grid[:, :] == self.GridLegend.FREE)  # returns are row and cols in a tuple
        obstacle_spaces = np.where(self.grid[:, :] == self.GridLegend.OBSTACLE)  # returns are row and cols in a tuple
        row = free_spaces[0]
        col = free_spaces[1]
        grid = deepcopy(self.grid)  # Copy over current position of grid (immediately after reset)
        grid_backup = deepcopy(self.grid)  # Backup for resetting the agent position
        v_values = np.zeros_like(grid)  # Obstacles will have a large negative value
        agent = self.agents[agent_id]

        for ii in range(len(row)):
            n = row[ii]
            m = col[ii]

            grid[n, m] = self.GridLegend.AGENT  # Set agent current position

            observation = torch.FloatTensor(grid).view(1, -1).to(agent.device)
            with torch.no_grad():
                # Compute soft Action-Value Q values
                q_values = agent.policy_model(observation)
                # Compute soft-Value V value of the agent being in that position
                value = agent.alpha * torch.logsumexp(q_values / agent.alpha, dim=1, keepdim=True)
                v_values[n, m] = value.cpu().detach().numpy()

            grid = deepcopy(grid_backup)  # Reset grid (erase agent current position)

        # Make sure the goal state is a sink (largest V value)
        v_values[agent.goal] = np.amax(v_values) + 0.5

        # Setting the obstacle V values to the lowest
        row = obstacle_spaces[0]
        col = obstacle_spaces[1]
        min_ = deepcopy(np.amin(v_values) - 0.5)

        for ii in range(len(row)):
            n = row[ii]
            m = col[ii]

            v_values[n, m] = min_

        # # Normalize V values
        # v_values = v_values/np.amax(v_values)

        return v_values

    def greedy_det_policy(self, v_values, agent_id):
        """Given a value function for an agent, the purpose of this function
        is to create a deterministic policy from any position to continue mission"""

        (rows, cols) = np.shape(self.grid)

        # Initializing arrow vectors for the quiver plot
        u = np.zeros((rows, cols))
        v = np.zeros((rows, cols))

        for n in range(rows):
            for m in range(cols):

                # Do not consider positions where the obstacles/terminal state are
                if self.grid[n, m] == self.GridLegend.OBSTACLE or (n, m) == self.agents[agent_id].goal:
                    continue

                moves = {}

                # Check above
                if n > 0:
                    moves[self.LegalActions.up] = v_values[n - 1, m]
                else:
                    moves[self.LegalActions.up] = np.amin(v_values)

                # Check below
                if n < rows - 1:
                    moves[self.LegalActions.down] = v_values[n + 1, m]
                else:
                    moves[self.LegalActions.down] = np.amin(v_values)  # Equal to obstacles' V values

                # Check left
                if m > 0:
                    moves[self.LegalActions.left] = v_values[n, m - 1]
                else:
                    moves[self.LegalActions.left] = np.amin(v_values)

                # Check right
                if m < cols - 1:
                    moves[self.LegalActions.right] = v_values[n, m + 1]
                else:
                    moves[self.LegalActions.right] = np.amin(v_values)

                action = max(moves, key=moves.get)

                if action == self.LegalActions.up:
                    u[n, m] = 0
                    v[n, m] = 1

                elif action == self.LegalActions.down:
                    u[n, m] = 0
                    v[n, m] = -1

                elif action == self.LegalActions.right:
                    u[n, m] = 1
                    v[n, m] = 0

                elif action == self.LegalActions.left:
                    u[n, m] = -1
                    v[n, m] = 0

        return u, v

    def render(self, mode='human', value_func=False, v_values=np.zeros((5, 5)),
               policy=False, u=np.zeros((5, 5)), v=np.zeros((5, 5))):

        # With or without greedy deterministic policy
        if not value_func:

            canvas = self.grid.copy()

            for agent in self.agents:
                # mark the visited cells in 0.6 gray
                # canvas[tuple(self.agents_visited_cells[agent])] = 0.6

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

                ax.imshow(canvas, interpolation='none', cmap='gray')

            else:
                super(GridWorld, self).render(mode=mode)  # just raise an exception for not Implemented mode

        elif value_func:

            # v_values_sq = v_values ** 2  # Squaring to see more contrast between places

            v_min = np.amin(v_values)
            v_max = np.amax(v_values)

            plt.imshow(v_values, vmin=v_min, vmax=v_max, zorder=0)

            plt.colorbar()
            plt.yticks(np.arange(-0.5, np.shape(self.grid)[0] + 0.5, step=1))
            plt.xticks(np.arange(-0.5, np.shape(self.grid)[1] + 0.5, step=1))
            plt.grid()

            if not policy:
                plt.title("Value function map using SQN")

        if policy:

            plt.quiver(np.arange(np.shape(self.grid)[0]), np.arange(np.shape(self.grid)[1]), u, v, zorder=10,
                       label="Policy")
            plt.title('Equivalent Greedy Policy')

        # plt.show()

    def seed(self, seed=None):
        """Sets the seed for the environment to maintain consistency during training"""

        rn_gen, seed = seeding.np_random(seed)

        return rn_gen, seed

    def grid2tensor(self, grid, agent):
        """Function to convert the observation into a n-dimensional grid according to the dict. size
         such that each grid has 1 at the position of the corresponding dict item. Needed for Conv_SQN"""

        key_grids = []

        for key in self.GridLegend:
            idx = np.where(grid == key.value)
            key_grid = np.zeros(grid.shape)
            key_grid[idx] = 1
            key_grids.append(key_grid)

        obs = torch.Tensor(key_grids)
        obs = obs.reshape(1, len(self.GridLegend), grid.shape[0], grid.shape[1]).to(agent.device)

        return obs
