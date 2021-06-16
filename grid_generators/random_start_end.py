import numpy as np
import random


def random_start_end(width=10, start_bounds=None, goal_bounds=None):
    """return a random start position in start_bounds and a random goal position in goal_bounds (can be the same
    values)
    :param width: width of the grid
    :param start_bounds: a tuple of tuples ((x0, x1), (y0, y1))
    :param goal_bounds: a tuple of tuples ((x0, x1), (y0, y1))
    :return: random start and goal coordinates"""

    if start_bounds is None:
        (start_x_bounds, start_y_bounds) = (0, width), (0, width)
    else:
        (start_x_bounds, start_y_bounds) = start_bounds
    if goal_bounds is None:
        (goal_x_bounds, goal_y_bounds) = (0, width), (0, width)
    else:
        (goal_x_bounds, goal_y_bounds) = goal_bounds

    start = random.randrange(*start_x_bounds), random.randrange(*start_y_bounds)
    goal = random.randrange(*goal_x_bounds), random.randrange(*goal_y_bounds)

    return start, goal
