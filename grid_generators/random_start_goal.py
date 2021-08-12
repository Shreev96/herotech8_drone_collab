import random


def random_start_goal(width=10, start_bounds=None, goal_bounds=None):
    """Return a random distinct start position in start_bounds and a random goal position in goal_bounds

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

    while goal == start:
        goal = random.randrange(*goal_x_bounds), random.randrange(*goal_y_bounds)

    return start, goal


def random_starts_goals(n, width=10, start_bounds=None, goal_bounds=None):
    """Return a random list of distinct start positions in start_bounds and a random goal positions in goal_bounds

        :param width: width of the grid
        :param start_bounds: a tuple of tuples ((x0, x1), (y0, y1))
        :param goal_bounds: a tuple of tuples ((x0, x1), (y0, y1))
        :return: random list of starts and goals coordinates"""

    if start_bounds is None:
        (start_x_bounds, start_y_bounds) = (0, width), (0, width)
    else:
        (start_x_bounds, start_y_bounds) = start_bounds
    if goal_bounds is None:
        (goal_x_bounds, goal_y_bounds) = (0, width), (0, width)
    else:
        (goal_x_bounds, goal_y_bounds) = goal_bounds

    starts = set()
    while len(starts) < n:
        # generate the starts positions
        starts.add((random.randrange(*start_x_bounds), random.randrange(*start_y_bounds)))

    goals = set()
    while len(goals) < n:
        # generate the goal positions
        goal = (random.randrange(*goal_x_bounds), random.randrange(*goal_y_bounds))
        if goal not in starts:
            goals.add(goal)

    return list(starts), list(goals)


def random_start_goal_in_subsquare(width=10, sub_width=10):
    """Place a random subsquare within a grid and return a random distinct start and goal positions within this square"

        :param width: width of the grid
        :param sub_width: width of the square within the grid
        :return: random start and goal coordinates"""
        
    assert width >= sub_width

    (top_x, top_y) = (random.randrange(0, width-sub_width+1), random.randrange(0, width-sub_width+1))

    _start_bounds = ((top_x, top_x + sub_width), (top_y, top_y + sub_width))
    _goal_bounds = ((top_x, top_x + sub_width), (top_y, top_y + sub_width))

    return random_start_goal(width=width, start_bounds=_start_bounds, goal_bounds=_goal_bounds)

def random_starts_goals_in_subsquare(n, width=10, sub_width=10):
    """Place a random subsquare within a grid and return a random distinct start and goal positions within this square"

        :param n: number of start/goal couple to create
        :param width: width of the grid
        :param sub_width: width of the square within the grid
        :return: random start and goal coordinates"""

    assert n*2 <= sub_width*sub_width, f"can't place n distincts starts and goals in a sub square of size {sub_width}"

    (top_x, top_y) = (random.randrange(0, width-sub_width+1), random.randrange(0, width-sub_width+1))
    _start_bounds = ((top_x, top_x + sub_width), (top_y, top_y + sub_width))
    _goal_bounds = ((top_x, top_x + sub_width), (top_y, top_y + sub_width))

    starts = set()
    while len(starts) < n:
        starts.add((random.randrange(*_start_bounds[0]), random.randrange(*_start_bounds[1])))
    
    goals = set()
    while len(goals) < n:
        goal = (random.randrange(*_goal_bounds[0]), random.randrange(*_goal_bounds[1]))
        if goal not in starts:
            goals.add(goal)
    
    return list(starts), list(goals)