import numpy as np


def find_neighbours(origin, max_y, max_x, min_x=0, min_y=0):
    """Given a position on the grid, this function is used to determine its neighbours"""

    neighbours = []

    (y, x) = origin

    # Check above
    if y > min_y:
        neighbours.append((y - 1, x))

    # Check below
    if y < max_y - 1:
        neighbours.append((y + 1, x))

    # Check left
    if x > min_x:
        neighbours.append((y, x - 1))

    # Check right
    if x < max_x - 1:
        neighbours.append((y, x + 1))

    return neighbours


def random_maze(start, goal, width=10, height=10, complexity=.75, density=.75):
    """Generate a random maze array.

    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``0`` and for free space is ``1``. ** This has been changed from the source.

    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm &
    https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
    """

    shape = (height, width)  # Shape of the grid

    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (shape[0] + shape[1])/2)
    density = int(density * ((shape[0] // 3) * (shape[1] // 3)))

    # Finding the neighbours of the goal state and the start state (can be anywhere)
    neighbours_start = find_neighbours(start, height, width)
    neighbours_end = find_neighbours(goal, height, width)

    neighbours_ = neighbours_start + neighbours_end  # Joining both lists

    while True:

        # Build actual maze
        Z = np.ones(shape, dtype=bool)

        # Make aisles
        for i in range(density):

            y, x = np.random.randint(2, shape[0] - 3), np.random.randint(2, shape[1] - 3)

            Z[y, x] = 0

            for j in range(complexity):
                neighbours = find_neighbours((y, x), height, width)

                if len(neighbours):
                    y_, x_ = neighbours[np.random.randint(0, len(neighbours))]

                    if Z[y_, x_] == 1:
                        Z[y_, x_] = 0
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 0
                        x, y = x_, y_

        # Start state, goal state and at least one neighbour of each must be free
        if Z[goal] == 1 and Z[start] == 1 and any(Z[neighbour] == 1 for neighbour in neighbours_):
            break

    grid = Z.astype(float)
    return grid
