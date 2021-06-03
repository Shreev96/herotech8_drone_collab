import math
import time

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
    episodes = 1000
    steps_done = 0
    target_update = 10
    agent_1 = AgentSQN(2, (0, 0), (4, 4), 5, device)
    # init_grid = random_maze(width=6, height=6, complexity=1, density=1)
    init_grid = np.array([
        [1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1.],
    ])

    # # Ordinary gridworld
    # env = GridWorld(agents=[agent_1], grid=init_grid)

    # Stochastic windy gridworld
    wind = [0, 1, 1, 0, 0]  # Row-wise wind addition towards the right (increasing col. idx)
    range_random_wind = 1
    probabilities = [0.2, 0.3, 0.5]  # [-1, 0, 1] noise added to wind-related position updates

    env = GridWorld(agents=[agent_1], grid=init_grid, col_wind=wind,
                    range_random_wind=range_random_wind, probabilities=probabilities)

    plt.ion()

    env.reset()
    env.render()

    start_time = time.time()
    for episode in range(episodes):
        obs = env.reset()

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
                agent.add_to_buffer(obs[i], action, reward, new_obs[i], agent.done)

            # move to the next state
            obs = new_obs

            # Perform one step of the optimisation
            for agent in env.agents:
                agent.train()

            if done:
                print(f"Episode {episode} finished after {step + 1} time steps")
                break

        steps_done += 1

    print("Complete")
    print("--- %s seconds ---" % (time.time() - start_time))

    # render result
    obs = env.reset()

    # evaluate value function for agent 1 (id = 0) and display
    v_values = env.eval_value_func(0)
    env.render(value_func=True, v_values=v_values)

    # devise a greedy deterministic policy for agent 1 and display
    U, V = env.greedy_det_policy(v_values, 0)  # U and V are x and y unit vectors respectively
    env.render(policy=True, u=U, v=V)

    for step in range(steps):
        env.render()
        actions = [env.agents[i].select_action(obs[i]) for i in range(len(env.agents))]
        obs, rewards, done, info = env.step(actions)
        if done:
            print(f"Episode final finished after {step + 1} time steps")
            break
    env.render()

    env.close()
    plt.ioff()
    plt.show()
