import math
import os
import time

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from SQN import AgentSQN
from DQN_Conv import AgentDQN
from gridworld import GridWorld
from random_grid_generator import random_maze

if __name__ == '__main__':
    # Model training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU
    # device = None
    steps = 100
    episodes = 500
    target_update = 10
    start_position = (0, 0)
    # init_grid = np.array([
    #     [1., 1., 0., 1., 1.],
    #     [1., 1., 1., 1., 0.],
    #     [1., 1., 1., 1., 1.],
    #     [0., 1., 1., 1., 1.],
    #     [1., 1., 0., 1., 1.],
    # ])

    # init_grid = np.genfromtxt('sample_grid/init_grid.csv', delimiter=',')  # generate a grid from a csv file
    init_grid = random_maze(width=6, height=6, complexity=1, density=1)
    goal = (len(init_grid) - 1, len(init_grid) - 1)  # place goal in bottom-right corner

    # create Agent : giving a start or goal position fix it, otherwise it is randomly generated at each reset
    agent_1 = AgentSQN(2, window_size=init_grid.shape[0], device=device, start=start_position, goal=goal)
    # agent_1 = AgentSQN(2, window_size=5, device=device)
    # agent_1 = AgentSQN(2, window_size=5, device=device, start=start_position)
    # agent_1 = AgentSQN(2, window_size=5, device=device, goal=goal)

    # # Ordinary gridworld
    env = GridWorld(agents=[agent_1], grid=init_grid)

    # Stochastic windy gridworld
    wind = [0, 1, 1, 0, 0]  # Row-wise wind addition towards the right (increasing col. idx)
    range_random_wind = 1
    probabilities = [0.2, 0.3, 0.5]  # [-1, 0, 1] noise added to wind-related position updates

    # env = GridWorld(agents=[agent_1], grid=init_grid, col_wind=wind,
    #                 range_random_wind=range_random_wind, probabilities=probabilities)

    # plt.ion()

    env.reset()
    env.render()
    plt.show()

    cum_rewards = []
    total_steps = []

    start_time = time.time()
    for episode in range(episodes):
        obs = env.reset()
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
                reward = torch.tensor([rewards[i]], device=device)
                cum_reward += reward.item()
                agent.add_to_buffer(obs[i], action, reward, new_obs[i], agent.done)

            # move to the next state
            obs = new_obs

            # Perform one step of the optimisation
            for agent in env.agents:
                agent.train()

            if done:
                break

        print(f"Episode {episode} finished after {step + 1} time steps")
        cum_rewards.append(cum_reward)
        total_steps.append(step)

    print("Complete")
    print("--- %s seconds ---" % (time.time() - start_time))
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
    # render result
    obs = env.reset()

    # # evaluate value function for agent 1 (id = 0) and display
    # v_values = env.eval_value_func(0)
    # env.render(value_func=True, v_values=v_values)
    # plt.show()

    # devise a greedy deterministic policy for agent 1 and display
    # U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
    # env.render(policy=True, u=U, v=V)
    # plt.show()


    # create gif of the result
    filenames = []
    plt.ion()
    for step in range(steps):
        env.render()
        plt.savefig(f'images/gif_frame/{step}.png')
        plt.show()

        filenames.append(f'images/gif_frame/{step}.png')
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
