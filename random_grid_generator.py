import numpy as np


def random_maze(width=81, height=51, complexity=.75, density=.75):
    """Generate a random maze array.

    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``0`` and for free space is ``1``. ** This has been changed from the source.

    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm &
    https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
    """

    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)

    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

    # Build actual maze
    Z = np.ones(shape, dtype=bool)

    # Fill borders
    Z[0, :] = Z[-1, :] = 0
    Z[:, 0] = Z[:, -1] = 0

    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1] // 2 + 1) * 2, np.random.randint(0, shape[0] // 2 + 1) * 2
        Z[y, x] = 0

        for j in range(complexity):
            neighbours = []

            if x > 1:
                neighbours.append((y, x - 2))

            if x < shape[1] - 2:
                neighbours.append((y, x + 2))

            if y > 1:
                neighbours.append((y - 2, x))

            if y < shape[0] - 2:
                neighbours.append((y + 2, x))

            if len(neighbours):
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]

                if Z[y_, x_] == 1:
                    Z[y_, x_] = 0
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 0
                    x, y = x_, y_

    # # Trimming off the borders
    grid = Z.astype(float)
    grid = grid[1:width, 1:height]

    return grid


# # Example use
# init_grid = random_maze(width=6, height=6, complexity=0.50, density=0.50)
# print(init_grid)
