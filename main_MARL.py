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
from torch.utils.tensorboard.summary import hparams

from DQN_Conv import AgentDQN
from SQN import AgentSQN, CoordinatorSQN
from grid_generators.random_start_goal import random_start_goal
from gridworld import GridWorld

# Model training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU


class SummaryWriter(SummaryWriter):
    def add_hparams(
            self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


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


def read_agents_config(config, env):
    n = config.getint("Agents Parameters", "n")
    model = config.get("Agents Parameters", "model")
    grid_size = config.getint("Grid Parameters", "grid_size")

    coordinator = None

    if model == "SQN":
        agents = [AgentSQN(i, obs_shape=env.observation_space.shape, device=DEVICE, config=config) for i in range(n)]
        coordinator = CoordinatorSQN(agents, env.observation_space.shape, DEVICE, config)
    elif model == "DQN":
        agents = [AgentDQN(i, obs_shape=env.observation_space.shape, device=DEVICE, config=config) for i in range(n)]
    else:
        raise NotImplementedError
    return agents, coordinator, model


def read_env_config(config):
    # Environmental conditions to be considered
    env_parameters = config['Environmental Effects']
    col_wind = env_parameters.getint("col_wind")
    range_random_wind = env_parameters.getint("range_random_wind")
    probabilities = env_parameters.getint("probabilities")

    return col_wind, range_random_wind, probabilities


def create_gif(filename, env, agents, reset_start_goal=True, reset_grid=True, steps=100, episodes=1):
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
            actions = [agents[i].select_action(obs[i]) for i in range(len(agents))]
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


def create_hparams(config):
    hyperparameters = {
        "steps": config["RL Parameters"]["steps"],
        "episodes": config["RL Parameters"]["episodes"],
        "train_period": config["RL Parameters"]["train_period"],
        "start_goal_reset_period": config["RL Parameters"]["start_goal_reset_period"],
        "grid_reset_period": config["RL Parameters"]["grid_reset_period"],

        "alpha": config["SQN Parameters"]["alpha"],
        "update_period": config["SQN Parameters"]["update_period"],
        "batch_size": config["SQN Parameters"]["batch_size"],

        # rewards:
        "FREE": config["Gridworld Parameters"]["FREE"],
        "GOAL": config["Gridworld Parameters"]["GOAL"],
        "OUT_OF_BOUNDS": config["Gridworld Parameters"]["OUT_OF_BOUNDS"],
        "OBSTACLES": config["Gridworld Parameters"]["OBSTACLE"]
    }
    return hyperparameters


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

    #######################
    # GRIDWORLDS CREATION #
    #######################
    n = config.getint("Agents Parameters", "n")

    if not effects:
        # Ordinary gridworld
        env = GridWorld(n_agents=n, grid=init_grid)

    else:
        # Stochastic windy gridworld
        col_wind, range_random_wind, probabilities = read_env_config(config)
        env = GridWorld(n_agents=n, grid=init_grid, col_wind=col_wind,
                        range_random_wind=range_random_wind, probabilities=probabilities)

    env.read_reward_config(config)

    ###################
    # AGENTS CREATION #
    ###################
    # create Agent : giving a start or goal position fix it, otherwise it is randomly generated at each reset
    # agent_1 = AgentSQN(2, window_size=init_grid.shape[0], device=device, start=start, goal=goal)
    # agent_1 = AgentSQN(2, window_size=grid_size, device=DEVICE)
    # agent_1 = AgentSQN(2, window_size=5, device=device, start=start_position)
    # agent_1 = AgentSQN(2, window_size=5, device=device, goal=goal)
    agents, coordinator, model = read_agents_config(config, env)

    ########
    # MAIN #
    ########
    # verification
    print("Hyperparameters :")
    print("steps", steps)
    print("episodes", episodes)
    print("train_period", train_period)
    print("start_goal_reset_period", start_goal_reset_period)
    print("grid_reset_period", grid_reset_period)
    print("alpha", agents[0].alpha)
    print("update_period", agents[0].update_steps)
    print("batch_size", agents[0].batch_size)
    print("reward free", env.rewards["free"])
    print("reward goal", env.rewards["goal"])
    print("reward obstacle", env.rewards["obstacles"])

    now = datetime.now().strftime('%Y%m%d%H%M%S')
    # Tensorboard initialisation (for logging values)
    log_dir = os.path.join("/content/runs", now)
    comment = create_comment(config)
    tb = SummaryWriter(log_dir=log_dir, comment=comment)

    plt.ion()

    total_reward_s = []
    cumulated_reward = np.zeros(len(agents))
    cumulated_rewards = []
    total_steps = []
    total_loss_s = []
    print("Start")
    try:
        # env.agents[0].init_pos = (0,2)
        # env.agents[1].init_pos = (0,7)
        # env.agents[0].init_goal = (9,2)
        # env.agents[1].init_goal = (9,7)
        env.reset(reset_starts_goals=True, reset_grid=False)  # put True if you want a random init grid
        env.render()
        plt.savefig(f"images/{now}.png")
        plt.show()

        train_period_s = np.linspace(start=2000, stop=train_period, num=50, endpoint=True, dtype=int)
        train_period_delay_s = np.linspace(start=0, stop=8000, num=50, endpoint=True, dtype=int)

        radius_s = np.linspace(start=2, stop=10, num=9, endpoint=True, dtype=int)
        radius_delays = np.linspace(start=0, stop=20000, num=9, endpoint=True, dtype=int)
        radius_index = 0
        radius = radius_s[0]

        # train_period_index = 0
        # train_period = train_period_s[0]

        reset_start_goal = True 
        reset_grid = False

        start_time = time.time()

        for episode in range(episodes):
            # if episode <=8000 and episode == train_period_delay_s[train_period_index]:
            #     train_period = train_period_s[train_period_index]
            #     train_period_index += 1
            #     print(f"New training period is {train_period} steps")

            # if episode <= 10000 and episode == radius_delays[radius_index]:
            #     radius = radius_s[radius_index]
            #     radius_index += 1
            #     print(f"Radius increased to {radius} cells")

            # if start_goal_period elapsed: change start and goal
            reset_start_goal = episode > 0 and episode % start_goal_reset_period == 0

            # if reset_grid_period elapsed: change grid
            # reset_grid = episode > 0 and episode % grid_reset_period == 0

            obs = env.reset(reset_starts_goals=reset_start_goal, radius=grid_size, reset_grid=reset_grid)

            working_agents = set(agents)  # set of agents with on-going mission

            total_reward = np.zeros(len(agents))
            total_loss = 0

            for step in range(steps):
                # env.render()

                # create new observations using the local obs of each agent and adding the pos and goal of other agents
                obs = [torch.cat([obs[i]] + [obs[j][:, -2:] for j in range(len(agents)) if j != i], dim=1)
                       for i in range(len(agents))]

                # select and perform actions
                actions = [agents[i].select_action(obs[i]) for i in range(len(agents))]
                new_obs, rewards, done, info = env.step(actions)

                # store the transition in memory
                for i in range(len(agents)):
                    agent = agents[i]
                    if agent in working_agents:
                        # only add relevant transitions: if agent mission was already done, don't add transition in replay memory
                        action = actions[i]
                        reward = torch.tensor([rewards[i]], device=DEVICE)
                        total_reward[i] += reward.item()
                        # compute new_observation the same way as before
                        new_observation = torch.cat([new_obs[i]] + [new_obs[j][:, -2:] for j in range(len(agents)) if j != i],
                                                    dim=1)

                        coordinator.add_to_buffer(obs[i], action, reward, new_observation, done[i])

                        if done[i]:
                            # if agent mission is done, remove it from working agents (to prevent adding future transitions)
                            print(f"Agent {i} is done after {step + 1} time steps")
                            working_agents.remove(agent)

                # move to the next state
                obs = new_obs

                # Perform one step of the optimisation
                if episode > 300 and step_done > 0 and step_done % train_period == 0:
                    loss = coordinator.train()
                    total_loss += loss * coordinator.batch_size

                step_done += 1
                if all(done):
                    break

            # # Perform one step of the optimisation
            # if episode > 0 and episode % train_period == 0:
            #     for i in range(len(agents)):
            #         loss = coordinator.train()
            #         total_loss += loss * coordinator.batch_size

            print(f"Episode {episode} finished after {step + 1} time steps")

            total_reward_s.append(total_reward)
            total_steps.append(step + 1)
            total_loss_s.append(total_loss)
            cumulated_reward += np.array(total_reward)
            cumulated_rewards.append(cumulated_reward.copy())

            # Tensorboard logging

            # for agent in agents:
            #     suffix = f"agent_{agent.id}"

            #     # agent target network parameters
            #     for name, weight in agent.target_model.named_parameters():
            #         tb.add_histogram(f"{suffix}.{name}", weight, episode)
            #         # tb.add_histogram(f"{suffix}.{name}.grad", weight.grad, episode)

            tb.add_scalar("Loss", total_loss, episode)
            tb.add_scalar("Total Reward", total_reward[0], episode)
            tb.add_scalar("Cumulated Reward", cumulated_reward[0], episode)
            tb.add_scalar("Total steps", step + 1, episode)

        print("Complete")
        print("--- %s seconds ---" % (time.time() - start_time))
    except Exception as e:
        raise e
    finally:
        total_reward_s = np.array(total_reward_s)
        cumulated_rewards = np.array(cumulated_rewards)

        # save models
        for agent in agents:
            agent.save(f"logs/models/{now}_{agent.id}.pt")

        # save config file in logs with good datetime
        with open(f"logs/configs/{now}.ini", "w") as configfile:
            config.write(configfile)

        data = {"Loss": total_loss_s,
                "Total Steps": total_steps}
        for i in range(len(agents)):
            data[f"Total Reward per episode {i}"] = total_reward_s[:, i]

        training_data = pd.DataFrame(data=data)
        training_data.to_csv(f"logs/csv/{now}.csv")

        # save cumulated reward plot
        plt.clf()
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
        fig.suptitle(("Cumulated Reward per episode for " + model))
        for i in range(len(agents)):
            ax[i].set_title(f"Agent {i}")
            ax[i].plot(total_reward_s[:,i])
        plt.xlabel("Epochs")
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

        # plt.figure()
        # # evaluate value function for agent 1 (id = 0) and display
        # v_values = env.eval_value_func(0, model)
        # env.render(value_func=True, v_values=v_values)
        # tb.add_figure("Value function", plt.gcf())
        # plt.show()
        #
        # # devise a greedy deterministic policy for agent 1 and display
        # U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
        # env.render(policy=True, u=U, v=V)
        # tb.add_figure("Greedy deterministic policy", plt.gcf())
        # plt.show()

        hyperparameters = create_hparams(config)
        metric_dict_ = {
            "Success Rate": improvement[-1] / len(total_steps),
            "Mean Cumulated Reward": cumulated_reward[0] / len(total_steps),
            "Mean Total Steps": np.mean(total_steps)
        }
        tb.add_hparams(hyperparameters, metric_dict=metric_dict_, run_name="")

    # create_gif(config_file[5:-4], env, agents, False, False, steps=100, episodes=1)

    tb.close()
    env.close()


if __name__ == '__main__':
    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config10MARL.ini")
    main(config)
