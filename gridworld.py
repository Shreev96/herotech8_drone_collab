from enum import IntEnum

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from rendering import fill_coords, point_in_rect, highlight_img, downsample


class WorldObj:
    """
    Base class for grid world objects
    """
    def __init__(self):
        self.pos = None

    def encode(self):
        """Encode the description of this object"""
        return None

    def on_entering(self, agent):
        """Action to perform when an agent enter this object"""
        raise NotImplementedError

    def on_leaving(self, agent):
        """Action to perform when an agent exit this object"""
        raise NotImplementedError

    def can_overlap(self):
        """Can an agent overlap this object?"""
        return True

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-rendered tiles
    tile_cache = {}

    def __init__(self, width, height):

        self.width = width
        self.height = height

        self.grid = np.empty(shape=(width, height), dtype=WorldObj)

    @classmethod
    def from_array(cls, array: np.ndarray):
        (width, height) = array.shape
        out = cls(width, height)
        out.grid = array
        return out

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

        return Grid.from_array(self.grid[top_x:(top_x+width), top_y:(top_y+height)])

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

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size=32, agent_pos=None, highlight_mask=None):
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

                agent_here = np.array_equal(agent_pos, (i, j))
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
        OUT_OF_BOUNDS = 6

    class UnknownAction(Exception):
        """Raised when an agent try to do an unknown action"""
        pass

    def __init__(self, agents=None, grid=np.ones((5, 5)), partial_obs=False, width=5, height=5, max_steps=100):
        if agents is None:
            agents = []
        self.agents = agents
        self.grid = grid  # TODO: make it random later

        # Define if the agents use partial observation or global observation
        self.partial_obs = partial_obs
        if self.partial_obs:
            self.agent_view_width = width
            self.agent_view_height = height

        self.actions = GridWorld.LegalActions
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(low=0, high=1, shape=grid.shape, dtype='uint8')

        self.agents_initial_pos = [agent.pos for agent in self.agents]  # starting position of the agents on the grid
        # self.agents_visited_cells = {agent: [] for agent in self.agents}  # initialise agents visited cells lists

        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        for i in range(len(self.agents)):
            agent = self.agents[i]
            agent.pos = self.agents_initial_pos[i]
            agent.done = False
            # self.agents_visited_cells[agent] = []
        # TODO: make it random later

        self.step_count = 0
        # self.render()  # show the initial arrangement of the grid

        # return first observation
        return self.gen_obs()

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
        elif (self.grid[n, m] == GridWorld.GridLegend.OBSTACLE  # obstacles
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
        self.step_count += 1

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
                raise GridWorld.UnknownAction("Unknown action")

            moves[i] = (n, m)

        # compute rewards and apply moves if they are legal
        for i in range(len(self.agents)):
            # compute rewards and illegal assertions
            rewards[i], illegal = self._reward_agent(i, moves[i])

            # apply move if legal
            if not illegal:
                self.agents[i].pos = moves[i]

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
            x, y = agent.pos
            w, h = self.agent_view_width, self.agent_view_height
            sub_grid = np.full((w, h), GridWorld.GridLegend.OUT_OF_BOUNDS)

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

            # mark visited cells
            # canvas[tuple(self.agents_visited_cells[agent])] = GridWorld.GridLegend.VISITED

            # mark agents : here there is a risk of contradiction with visited cells
            # canvas[(agent.pos for agent in self.agents)] = GridWorld.GridLegend.AGENT
            for agent in self.agents:
                canvas[agent.pos] = agent.id

            return canvas

    def render(self, mode='human'):
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
            plt.show()
        else:
            super(GridWorld, self).render(mode=mode)  # just raise an exception for not Implemented mode
