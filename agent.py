import random
from collections import namedtuple, deque
from enum import IntEnum


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

    class GridLegend(IntEnum):
        """ Replica of the version in the gridworld class. Needed to create the Conv-net"""
        # FREE = 0
        AGENT = 2
        OBSTACLE = 1
        # VISITED = 5  # TODO: Commented now to help with creation of Convnet input
        # OUT_OF_BOUNDS = 6  # TODO: Commented now to help with creation of Convnet input
        GOAL = 7

    Transition = namedtuple('Transition',
                            ('state', 'action', 'reward', 'next_state', 'done'))

    def __init__(self, i, device, start=None, goal=None):
        # ID of the agent (represents the integer number to look for on the grid
        self.id = i

        self.grid_object = None

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
        self.experience_replay = deque([], maxlen=40000)

        # PyTorch stuff
        self.device = device

    def add_to_buffer(self, s, a, r, s_, done):
        """Save a transition in the experience replay memory"""

        # s = s.view(1, -1).float()
        # s_ = s_.view(1, -1).float()

        # Now, 'env.grid2tensor' produces tensor observations directly, so there is no need to cast
        s = s.float()  # convert to float because Conv_SQN expects float and not double
        s_ = s_.float()

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

    def train(self) -> float:
        """Perform a single step of the optimization and return the loss"""
        raise NotImplementedError("Must implement this method")

    def save(self, path):
        """Save the state of the model"""
        raise NotImplementedError("Must implement this method")

    def load(self, path):
        """Load the parameter of the model from a file"""
        raise NotImplementedError("Must implement this method")