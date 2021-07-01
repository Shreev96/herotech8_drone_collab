import os
import time
import configparser
from datetime import datetime

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from SQN import AgentSQN
from DQN_Conv import AgentDQN

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
    effects = rl_parameters.getboolean("effects")

    return steps, episodes, train_period, start_goal_reset_period, effects


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
        return [AgentSQN(i, window_size=grid_size, device=DEVICE) for i in range(n)], model
    elif model == "DQN":
        return [AgentDQN(i, window_size=grid_size, device=DEVICE) for i in range(n)], model
    else:
        raise NotImplementedError


def read_env_config(config_file):
    # Environmental conditions to be considered

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(config_file)

    env_parameters = config['Environmental Effects']
    col_wind = env_parameters.getint("col_wind")
    range_random_wind = env_parameters.getint("range_random_wind")
    probabilities = env_parameters.getint("probabilities")

    return col_wind, range_random_wind, probabilities


def create_gif(filename, env, init_grid, steps=100, episodes=1):
    """ render result and create gif of the result"""
    filenames = []
    fig = plt.figure()
    plt.ion()
    grid_size = len(env.grid)

    for episode in range(episodes):
        start, goal = random_start_end(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                       goal_bounds=((grid_size - 7, grid_size), (0, grid_size)))

        obs = env.reset(init_grid=init_grid, starts=[start], goals=[goal])
        for step in range(steps):
            env.render()
            plt.savefig(f'images/gif_frame/E{episode:03}S{step:05}.png')
            plt.cla()

            filenames.append(f'images/gif_frame/E{episode:03}S{step:05}.png')
            actions = [env.agents[i].select_action(obs[i]) for i in range(len(env.agents))]
            obs, rewards, done, info = env.step(actions)
            if done:
                break
        print(f"Episode finished after {step + 1} time steps")
    env.render()
    plt.savefig(f'images/gif_frame/final.png')
    plt.cla()
    filenames.append(f'images/gif_frame/final.png')
    plt.ioff()

    with imageio.get_writer(f'images/gif/{filename}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            # TODO : try to save fig as numpy array to preserve data and speed up the rendering

    for filename in set(filenames):
        os.remove(filename)


def create_gif2(filename, env, init_grid, steps=100, episodes=1):
    """ render result and create gif of the result"""
    fig = plt.figure()
    grid_size = len(env.grid)
    with imageio.get_writer(f'images/gif/{filename}.gif', mode='I') as writer:
        for episode in range(episodes):
            start, goal = random_start_end(width=grid_size, start_bounds=((0, grid_size // 2), (0, grid_size)),
                                           goal_bounds=((grid_size // 2, grid_size), (0, grid_size)))
            obs = env.reset(init_grid=init_grid, starts=[start], goals=[goal])
            for step in range(steps):
                env.render()
                fig.canvas.draw()
                data = np.asarray(fig.canvas.buffer_rgba())
                writer.append_data(data)

                actions = [env.agents[i].select_action(obs[i]) for i in range(len(env.agents))]
                obs, rewards, done, info = env.step(actions)
                if done:
                    break
            print(f"Episode finished after {step + 1} time steps")


def main(config_file):
    print(DEVICE)

    steps, episodes, train_period, start_goal_reset_period, effects = read_config(config_file)
    step_done = 0

    #################
    # GRID CREATION #
    #################
    grid_size = read_grid_config(config_file)  # Square grid size

    # init_grid = np.genfromtxt('sample_grid/init_grid.csv', delimiter=',')  # generate a grid from a csv file
    # init_grid = random_maze(start, goal, width=grid_size, height=grid_size, complexity=0.5, density=0.5)
    init_grid = np.ones((grid_size, grid_size))  # empty grid

    ####################
    # STARTS AND GOALS #
    ####################
    # start = (0, 0)
    # goal = (grid_size - 1, grid_size - 1)  # place goal in bottom-right corner
    start, goal = random_start_end(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                   goal_bounds=((grid_size - 7, grid_size), (0, grid_size)))

    ###################
    # AGENTS CREATION #
    ###################
    # create Agent : giving a start or goal position fix it, otherwise it is randomly generated at each reset
    # agent_1 = AgentSQN(2, window_size=init_grid.shape[0], device=device, start=start, goal=goal)
    # agent_1 = AgentSQN(2, window_size=grid_size, device=DEVICE)
    # agent_1 = AgentSQN(2, window_size=5, device=device, start=start_position)
    # agent_1 = AgentSQN(2, window_size=5, device=device, goal=goal)
    agents, model = read_agents_config(config_file)

    #######################
    # GRIDWORLDS CREATION #
    #######################
    if not effects:
        # Ordinary gridworld
        env = GridWorld(agents=agents, grid=init_grid)

    else:
        # Stochastic windy gridworld
        col_wind, range_random_wind, probabilities = read_env_config(config_file)
        env = GridWorld(agents=agents, grid=init_grid, col_wind=col_wind,
                        range_random_wind=range_random_wind, probabilities=probabilities)

    env.read_reward_config(config_file)

    ########
    # MAIN #
    ########

    plt.ion()

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
                start, goal = random_start_end(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                               goal_bounds=((grid_size - 7, grid_size), (0, grid_size)))
                print(f"New start is {start} and new goal is {goal}")

            cum_rewards.append(cum_reward)
            total_steps.append(step)

        print("Complete")
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        raise e
    finally:
        now = datetime.now()

        env.save(directory="logs/models", datetime=now.strftime('%Y%m%d%H%M%S'))

        plt.clf()
        plt.title(("Cumulated Reward for " + model + f"\n{config_file}"))
        plt.xlabel("Epochs")
        plt.plot(cum_rewards)
        plt.savefig(f"logs/cumulated_rewards/{now.strftime('%Y%m%d%H%M%S')}.png")
        plt.clf()

        plt.title(("Total steps for " + model + f"\n{config_file}"))
        plt.xlabel("Epochs")
        plt.plot(total_steps)
        plt.savefig(f"logs/total_steps/{now.strftime('%Y%m%d%H%M%S')}.png")
        plt.clf()

        # Calculate improvement over training duration
        # Equivalent to the # of times the agent reaches the goal within the designated number of timesteps
        improvement = np.zeros_like(total_steps)
        counter = 0
        for ii in range(len(total_steps)):
            if total_steps[ii] < steps - 1:
                counter += 1
            improvement[ii] = counter

        plt.title("Improvement over training episodes for " + model + f"\n{config_file}")
        plt.xlabel("Epochs")
        plt.plot(improvement)
        plt.savefig(f"logs/improvement/{now.strftime('%Y%m%d%H%M%S')}.png")
        plt.clf()

        plt.figure()
        # evaluate value function for agent 1 (id = 0) and display
        v_values = env.eval_value_func(0, model)
        env.render(value_func=True, v_values=v_values)
        plt.show()

        # devise a greedy deterministic policy for agent 1 and display
        U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
        env.render(policy=True, u=U, v=V)
        plt.show()

    create_gif("G10S100E5", env, init_grid, steps=100, episodes=5)

    env.close()


if __name__ == '__main__':
    main("config10.ini")
