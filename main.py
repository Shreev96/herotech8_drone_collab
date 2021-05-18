import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from gridworld import GridWorld

if __name__ == '__main__':
    # Model training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU

    steps = 500
    episodes = 1000
    steps_done = 0
    target_update = 10
    agent_1 = Agent(2, (0, 0), (4, 4), 5, device)
    init_grid = np.array([
        [1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.],
        [0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1.],
    ])
    env = GridWorld(agents=[agent_1], grid=init_grid)
    plt.ion()

    env.reset()
    env.render()

    for episode in range(episodes):
        obs = env.reset()

        agent_1.eps = 0.05 + (0.9 - 0.05) * math.exp(-1 * steps_done / 1000)
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
                agent.add_to_buffer(obs[i], action, reward, new_obs[i])

            # move to the next state
            obs = new_obs

            # Perform one step of the optimisation
            for agent in env.agents:
                agent.train()

            if done:
                print(f"Episode {episode} finished after {step + 1} time steps")
                break
            # Update the target network, copying all weights and biases in DQN

        if episode % target_update == 0:
            # Update target network for each of the agents
            for i in range(len(env.agents)):
                agent = env.agents[i]
                agent.target_model.load_state_dict(
                    agent.policy_model.state_dict()
                                                  )
        steps_done += 1

    print("Complete")

    # render result
    obs = env.reset()
    agent_1.eps = 0
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