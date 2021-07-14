import configparser
import os
import time
from datetime import datetime

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from DQN_Conv import AgentDQN
from SQN import AgentSQN
from grid_generators.random_start_goal import random_start_goal
from gridworld import GridWorld

# Model training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU


def read_config(config):
    rl_parameters = config['RL Parameters']
    steps = rl_parameters.getint("steps")
    episodes = rl_parameters.getint("episodes")
    train_period = rl_parameters.getint("train_period")
    start_goal_reset_period = rl_parameters.getint("start_goal_reset_period")
    grid_reset_period = rl_parameters.getint("grid_reset_period")
    effects = rl_parameters.getboolean("effects")

    return steps, episodes, train_period, start_goal_reset_period, grid_reset_period, effects


def read_grid_config(config):
    return config.getint("Grid Parameters", "grid_size")


def read_agents_config(config):
    n = config.getint("Agents Parameters", "n")
    model = config.get("Agents Parameters", "model")
    grid_size = config.getint("Grid Parameters", "grid_size")

    if model == "SQN":
        return [AgentSQN(i, window_size=grid_size, device=DEVICE) for i in range(n)], model
    elif model == "DQN":
        return [AgentDQN(i, window_size=grid_size, device=DEVICE) for i in range(n)], model
    else:
        raise NotImplementedError


def read_env_config(config):
    # Environmental conditions to be considered
    env_parameters = config['Environmental Effects']
    col_wind = env_parameters.getint("col_wind")
    range_random_wind = env_parameters.getint("range_random_wind")
    probabilities = env_parameters.getint("probabilities")

    return col_wind, range_random_wind, probabilities


def create_gif(filename, env, reset_start_goal=True, reset_grid=True, steps=100, episodes=1):
    """ render result and create gif of the result"""
    filenames = []
    fig = plt.figure()
    plt.ion()
    grid_size = len(env.grid)

    for episode in range(episodes):
        obs = env.reset(reset_start_goal, reset_grid)
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
            start, goal = random_start_goal(width=grid_size, start_bounds=((0, grid_size // 2), (0, grid_size)),
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


def create_comment(config):
    steps = config["RL Parameters"]["steps"]
    episodes = config["RL Parameters"]["episodes"]
    train_period = config["RL Parameters"]["train_period"]
    start_goal_reset_period = config["RL Parameters"]["start_goal_reset_period"]
    grid_reset_period = config["RL Parameters"]["grid_reset_period"]

    alpha = config["SQN Parameters"]["alpha"]
    update_period = config["SQN Parameters"]["update_period"]
    batch_size = config["SQN Parameters"]["batch_size"]

    free = config["Gridworld Parameters"]["FREE"]
    goal = config["Gridworld Parameters"]["GOAL"]
    out_of_b = config["Gridworld Parameters"]["OUT_OF_BOUNDS"]
    obstacle = config["Gridworld Parameters"]["OBSTACLE"]

    return f" steps={steps} episodes={episodes} train_period={train_period} start_goal_reset_period={start_goal_reset_period}" \
           f" grid_reset_period={grid_reset_period} alpha={alpha} update_period={update_period} batch_size={batch_size}" \
           f" rwd_free={free} rwd_goal={goal} rwd_out_of_b={out_of_b} rwd_obstacle={obstacle}"


def main(config: configparser.ConfigParser):
    print(DEVICE)

    steps, episodes, train_period, start_goal_reset_period, grid_reset_period, effects = read_config(config)
    step_done = 0

    #################
    # GRID CREATION #
    #################
    grid_size = read_grid_config(config)  # Square grid size

    # init_grid = np.genfromtxt('sample_grid/init_grid.csv', delimiter=',')  # generate a grid from a csv file
    # init_grid = random_maze(start, goal, width=grid_size, height=grid_size, complexity=0.5, density=0.5)
    init_grid = np.zeros((grid_size, grid_size))  # empty grid

    ####################
    # STARTS AND GOALS #
    ####################
    # start = (0, 0)
    # goal = (grid_size - 1, grid_size - 1)  # place goal in bottom-right corner
    start, goal = random_start_goal(width=grid_size, start_bounds=((0, 1), (0, grid_size)),
                                    goal_bounds=((grid_size - 7, grid_size), (0, grid_size)))

    ###################
    # AGENTS CREATION #
    ###################
    # create Agent : giving a start or goal position fix it, otherwise it is randomly generated at each reset
    # agent_1 = AgentSQN(2, window_size=init_grid.shape[0], device=device, start=start, goal=goal)
    # agent_1 = AgentSQN(2, window_size=grid_size, device=DEVICE)
    # agent_1 = AgentSQN(2, window_size=5, device=device, start=start_position)
    # agent_1 = AgentSQN(2, window_size=5, device=device, goal=goal)
    agents, model = read_agents_config(config)

    #######################
    # GRIDWORLDS CREATION #
    #######################
    if not effects:
        # Ordinary gridworld
        env = GridWorld(agents=agents, grid=init_grid)

    else:
        # Stochastic windy gridworld
        col_wind, range_random_wind, probabilities = read_env_config(config)
        env = GridWorld(agents=agents, grid=init_grid, col_wind=col_wind,
                        range_random_wind=range_random_wind, probabilities=probabilities)

    env.read_reward_config(config)

    ########
    # MAIN #
    ########

    # Tensorboard initialisation (for logging values)
    comment = create_comment(config)
    tb = SummaryWriter(comment=comment)

    plt.ion()

    cum_rewards = []
    total_steps = []
    total_loss_s = []

    try:
        env.reset()
        env.render()
        plt.show()

        train_period_s = np.linspace(start=5000, stop=train_period, num=200, endpoint=True, dtype=int)
        train_period_delay_s = np.linspace(start=0, stop=8000, num=200, endpoint=True, dtype=int)

        train_period_index = 0
        # train_period = train_period_s[0]

        reset_start_goal = True
        reset_grid = True

        start_time = time.time()

        for episode in range(episodes):
            # if episode <=8000 and episode == train_period_delay_s[train_period_index]:
            #     train_period = train_period_s[train_period_index]
            #     train_period_index += 1
            #     print(f"New training period is {train_period} steps")


            # if start_goal_period elapsed: change start and goal
            reset_start_goal = episode > 0 and episode % start_goal_reset_period == 0

            # if reset_grid_period elapsed: change grid
            reset_grid = episode > 0 and episode % grid_reset_period == 0

            obs = env.reset(reset_starts_goals=reset_start_goal, reset_grid=reset_grid)
            reset_start_goal = False
            reset_grid = False

            cum_reward = 0
            total_loss = [0 for agent in env.agents]

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
                if step_done > 10000 and step_done % train_period == 0:
                    for i in range(len(env.agents)):
                        agent = env.agents[i]
                        loss = agent.train()
                        total_loss[i] += loss * agent.batch_size

                step_done += 1
                if done:
                    break

            print(f"Episode {episode} finished after {step + 1} time steps")

            cum_rewards.append(cum_reward)
            total_steps.append(step + 1)
            total_loss_s.append(total_loss)

            # Tensorboard logging
            for agent in env.agents:
                suffix = f"agent_{agent.agent_id}"

                # agent target network parameters
                for name, weight in agent.target_model.named_parameters():
                    tb.add_histogram(f"{suffix}.{name}", weight, episode)
                    tb.add_histogram(f"{suffix}.{name}.grad", weight.grad, episode)

            tb.add_scalar("Loss", total_loss[0], episode)
            tb.add_scalar("Cumulated Reward", cum_reward, episode)
            tb.add_scalar("Total steps", step + 1, episode)

        print("Complete")
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        raise e
    finally:
        now = datetime.now().strftime('%Y%m%d%H%M%S')

        # save model
        env.save(directory="logs/models", datetime=now)

        # save config file in logs with good datetime
        with open(f"logs/configs/{now}.ini", "w") as configfile:
            config.write(configfile)

        training_data = pd.DataFrame(data=[cum_rewards, total_loss_s[:][0], total_steps],
                                     columns=["Cumulated Reward", "Loss", "Total Steps"])
        training_data.to_csv(f"logs/csv/{now}.csv")

        # save cumulated reward plot
        plt.clf()
        plt.title(("Cumulated Reward for " + model))
        plt.xlabel("Epochs")
        plt.plot(cum_rewards)
        plt.savefig(f"logs/cumulated_rewards/{now}.png")
        tb.add_figure("Cumulated Reward Plot", plt.gcf())
        plt.clf()

        # save total steps plot
        plt.title(("Total steps for " + model))
        plt.xlabel("Epochs")
        plt.plot(total_steps)
        plt.savefig(f"logs/total_steps/{now}.png")
        tb.add_figure("Total steps Plot", plt.gcf())
        plt.clf()

        # Calculate improvement over training duration
        # Equivalent to the # of times the agent reaches the goal within the designated number of timesteps
        improvement = np.zeros_like(total_steps)
        counter = 0
        for ii in range(len(total_steps)):
            if total_steps[ii] < steps - 1:
                counter += 1
            improvement[ii] = counter
            tb.add_scalar("Improvement", counter, ii)

        # save improvement plot
        plt.title("Improvement over training episodes for " + model)
        plt.xlabel("Epochs")
        plt.plot(improvement)
        plt.savefig(f"logs/improvement/{now}.png")
        tb.add_figure("Improvement Plot", plt.gcf())
        plt.clf()

        plt.figure()
        # evaluate value function for agent 1 (id = 0) and display
        v_values = env.eval_value_func(0, model)
        env.render(value_func=True, v_values=v_values)
        tb.add_figure("Value function", plt.gcf())
        plt.show()

        # devise a greedy deterministic policy for agent 1 and display
        U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
        env.render(policy=True, u=U, v=V)
        tb.add_figure("Greedy deterministic policy", plt.gcf())
        plt.show()

    # create_gif(config_file[5:-4], env, False, False, steps=100, episodes=1)

    tb.close()
    env.close()


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config10.ini")
    main(config)
