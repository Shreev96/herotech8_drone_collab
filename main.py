import os
import time
import configparser

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from SQN import AgentSQN
from gridworld import GridWorld
from grid_generators.random_grid_generator import random_maze
from grid_generators.random_start_end import random_start_end

# Model training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU


def read_config(config_file):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)

    rl_parameters = config['RL Parameters']
    steps = rl_parameters.getint("steps")
    episodes = rl_parameters.getint("episodes")
    train_period = rl_parameters.getint("train_period")
    start_goal_reset_period = rl_parameters.getint("start_goal_reset_period")

    return steps, episodes, train_period, start_goal_reset_period


def read_grid_config(config_file):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)
    return config.getint("Grid Parameters", "grid_size")


def read_agents_config(config_file):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)
    n = config.getint("Agents Parameters", "n")
    model = config.get("Agents Parameters", "model")
    grid_size = config.getint("Grid Parameters", "grid_size")

    if model == "SQN":
        return [AgentSQN(i, window_size=grid_size, device=DEVICE) for i in range(n)]
    else:
        raise NotImplementedError


if __name__ == '__main__':
    print(DEVICE)

    steps, episodes, train_period, start_goal_reset_period = read_config("config.ini")
    step_done = 0

    #################
    # GRID CREATION #
    #################
    grid_size = read_grid_config("config.ini")  # Square grid size

    # init_grid = np.genfromtxt('sample_grid/init_grid.csv', delimiter=',')  # generate a grid from a csv file
    # init_grid = random_maze(start, goal, width=grid_size, height=grid_size, complexity=0.5, density=0.5)
    init_grid = np.ones((grid_size, grid_size))  # empty grid

    ####################
    # STARTS AND GOALS #
    ####################
    # start = (0, 0)
    # goal = (grid_size - 1, grid_size - 1)  # place goal in bottom-right corner
    start, goal = random_start_end(width=grid_size, start_bounds=((0, grid_size//2), (0, grid_size)),
                                   goal_bounds=((grid_size//2, grid_size), (0, grid_size)))

    ###################
    # AGENTS CREATION #
    ###################
    # create Agent : giving a start or goal position fix it, otherwise it is randomly generated at each reset
    # agent_1 = AgentSQN(2, window_size=init_grid.shape[0], device=device, start=start, goal=goal)
    # agent_1 = AgentSQN(2, window_size=grid_size, device=DEVICE)
    # agent_1 = AgentSQN(2, window_size=5, device=device, start=start_position)
    # agent_1 = AgentSQN(2, window_size=5, device=device, goal=goal)
    agents = read_agents_config("config.ini")

    #######################
    # GRIDWORLDS CREATION #
    #######################
    # # Ordinary gridworld
    env = GridWorld(agents=agents, grid=init_grid)
    env.read_reward_config("config.ini")

    # # Stochastic windy gridworld
    # wind = [0, 1, 1, 0, 0]  # Row-wise wind addition towards the right (increasing col. idx)
    # range_random_wind = 1
    # probabilities = [0.2, 0.3, 0.5]  # [-1, 0, 1] noise added to wind-related position updates

    # env = GridWorld(agents=[agent_1], grid=init_grid, col_wind=wind,
    #                 range_random_wind=range_random_wind, probabilities=probabilities)

    ########
    # MAIN #
    ########

    # plt.ion()

    cum_rewards = []
    total_steps = []

    try:
        env.reset(init_grid=init_grid, starts=[start], goals=[goal])
        print(f"First start is {start} and first goal is {goal}")
        env.render()
        plt.show()


        start_time = time.time()
        for episode in range(episodes):
            obs = env.reset(init_grid=init_grid, starts=[start], goals=[goal])
            cum_reward = 0

            for step in range(steps):
                # env.render()
                # select and perform actions
                actions = [env.agents[i].select_action(obs[i]) for i in range(len(env.agents))]
                new_obs, rewards, done, info = env.step(actions)

                # store the transition in memory
                for i in range(len(env.agents)):
                    agent = env.agents[i]
                    action = actions[i]
                    reward = torch.tensor([rewards[i]], device=DEVICE)
                    cum_reward += reward.item()
                    agent.add_to_buffer(obs[i], action, reward, new_obs[i], agent.done)

                # move to the next state
                obs = new_obs

                # Perform one step of the optimisation
                if step_done % train_period == 0:
                    for agent in env.agents:
                        agent.train()

                step_done += 1
                if done:
                    break

            print(f"Episode {episode} finished after {step + 1} time steps")

            # if start_goal_period elapsed: change start and goal
            if episode > 0 and episode % start_goal_reset_period == 0:
                start, goal = random_start_end(width=grid_size, start_bounds=((0, grid_size // 2), (0, grid_size)),
                                               goal_bounds=((grid_size // 2, grid_size), (0, grid_size)))
                print(f"New start is {start} and new goal is {goal}")

            cum_rewards.append(cum_reward)
            total_steps.append(step)

        print("Complete")
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        raise e
    finally:
        env.save(directory="logs")

    reward_fig = plt.figure(1)
    plt.title("Cumulated Reward")
    plt.xlabel("Epochs")
    plt.plot(cum_rewards)
    plt.show()

    steps_fig = plt.figure(2)
    plt.title("Total steps")
    plt.xlabel("Epochs")
    plt.plot(total_steps)
    plt.show()

    plt.figure()
    # # evaluate value function for agent 1 (id = 0) and display
    # v_values = env.eval_value_func(0)
    # env.render(value_func=True, v_values=v_values)
    # plt.show()
    #
    # # devise a greedy deterministic policy for agent 1 and display
    # U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
    # env.render(policy=True, u=U, v=V)
    # plt.show()

    # render result
    # create gif of the result
    filenames = []
    plt.ion()
    for episode in range(1):
        start, goal = random_start_end(width=grid_size, start_bounds=((0, grid_size // 2), (0, grid_size)),
                                       goal_bounds=((grid_size // 2, grid_size), (0, grid_size)))
        obs = env.reset(init_grid=init_grid, starts=[start], goals=[goal])
        for step in range(steps):
            env.render()
            plt.savefig(f'images/gif_frame/E{episode:03}S{step:05}.png')
            plt.cla()

            filenames.append(f'images/gif_frame/E{episode:03}S{step:05}.png')
            actions = [env.agents[i].select_action(obs[i]) for i in range(len(env.agents))]
            obs, rewards, done, info = env.step(actions)
            if done:
                print(f"Episode final finished after {step + 1} time steps")
                break
    env.render()
    plt.savefig(f'images/gif_frame/final.png')
    plt.show()
    filenames.append(f'images/gif_frame/final.png')
    env.close()
    plt.ioff()

    with imageio.get_writer('images/gif/simple.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in set(filenames):
        os.remove(filename)
