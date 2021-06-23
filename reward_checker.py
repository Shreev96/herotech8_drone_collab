import matplotlib.pyplot as plt
import numpy as np
import torch

from grid_generators.random_start_end import random_start_end
from gridworld import GridWorld
from main import read_config, read_agents_config, read_grid_config, read_env_config

# Model training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU


if __name__ == '__main__':
    config_file = "config.ini"

    steps, episodes, train_period, start_goal_reset_period, effects = read_config(config_file)
    grid_size = read_grid_config(config_file)
    init_grid = np.ones((grid_size, grid_size))
    agents, model = read_agents_config(config_file)
    if not effects:
        # Ordinary gridworld
        env = GridWorld(agents=agents, grid=init_grid)

    else:
        # Stochastic windy gridworld
        col_wind, range_random_wind, probabilities = read_env_config(config_file)
        env = GridWorld(agents=agents, grid=init_grid, col_wind=col_wind,
                        range_random_wind=range_random_wind, probabilities=probabilities)

    env.read_reward_config(config_file)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    start, goal = random_start_end(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                   goal_bounds=((grid_size - 1, grid_size), (0, grid_size)))
    line, = ax2.plot([0])
    plt.subplot(121)

    reward = [0]

    def key_handler(event):
        fig.canvas.flush_events()
        action = None
        if event.key == "left":
            action = GridWorld.LegalActions.left

        if event.key == "right":
            action = GridWorld.LegalActions.right

        if event.key == "up":
            action = GridWorld.LegalActions.up

        if event.key == "down":
            action = GridWorld.LegalActions.down

        new_obs, rewards, done, info = env.step([action])
        reward.append(reward[-1] + rewards[0])
        ax2.cla()
        ax2.plot(reward)
        line.figure.canvas.draw_idle()
        if done:
            env.render()
            fig.canvas.draw()
            fig.canvas.flush_events()
            start, goal = random_start_end(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                           goal_bounds=((grid_size - 1, grid_size), (0, grid_size)))
            env.reset(init_grid=init_grid, starts=[start], goals=[goal])

        env.render()
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', key_handler)
    obs = env.reset(init_grid=init_grid, starts=[start], goals=[goal])
    env.render()
