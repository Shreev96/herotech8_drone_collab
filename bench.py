from datetime import datetime

from main_MARL import main
import configparser
import itertools

import os
import shutil

def bench1():
    hyperparameters = {
        "steps" : [100],
        "episodes" : [10000],
        "train_period" : [5],
        "start_goal_reset_period" : [1],
        "grid_reset_period" : [20],

        "alpha" : [0.05],
        "update_period" : [5],
        "batch_size" : [512],

        # rewards:
        "FREE" : [-0.01],
        "GOAL" : [10.0],
        # "OUT_OF_BOUNDS" : [-0.1],
        "OBSTACLES" : [-1.0]
    }

    params_values = [v for v in hyperparameters.values()]

    grid_size = 10

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config10MARL.ini")
    config["Grid Parameters"]["grid_size"] = str(grid_size)

    total_len = len(list(itertools.product(*params_values)))

    run_number = 0
    for (steps, episodes, train_period, start_goal_reset_period, grid_reset_period, 
        alpha, update_period, batch_size,
        free, goal, obstacle) in itertools.product(*params_values):

        run_number += 1
        config["RL Parameters"]["steps"] = str(steps)
        config["RL Parameters"]["episodes"] = str(episodes)
        config["RL Parameters"]["train_period"] = str(train_period)
        config["RL Parameters"]["start_goal_reset_period"] = str(start_goal_reset_period)
        config["RL Parameters"]["grid_reset_period"] = str(grid_reset_period)

        config["SQN Parameters"]["alpha"] = str(alpha)
        config["SQN Parameters"]["update_period"] = str(update_period)
        config["SQN Parameters"]["batch_size"] = str(batch_size)

        config["Gridworld Parameters"]["FREE"] = str(free)
        config["Gridworld Parameters"]["GOAL"] = str(goal)
        config["Gridworld Parameters"]["OUT_OF_BOUNDS"] = str(free)  # TODO : for now
        config["Gridworld Parameters"]["OBSTACLE"] = str(obstacle)

        print(f"Run {run_number} on {total_len}")
        main(config)
    print(f"{run_number} runs on {total_len} finished")


def bench2():

    grid_size = 10

    values = [
        (100, 1000, 3, 10, 0.01, 0.0, 1000.0, -10.0),
        (100, 1000, 10, 1, 0.01, 0.0, 1000.0, -10.0),
        (100, 1000, 3, 1, 10.0, -1.0, 1000.0, -10.0),
        (100, 1000, 3, 10, 0.01, -0.04, 1000.0, -10.0),
        (100, 1000, 10, 10, 1.0, -1.0, 1000.0, -10.0),
        (100, 1000, 3, 1, 1.0, 0.0, 1000.0, -10.0),
        (100, 1000, 3, 1, 1.0, -0.04, 1000.0, -10.0),
        (100, 1000, 10, 10, 1.0, -0.04, 1000.0, -10.0),
        (100, 1000, 3, 10, 10, -0.04, 1000.0, -10.0)
    ]

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config.ini")
    config["Grid Parameters"]["grid_size"] = str(grid_size)

    total_len = len(values)

    run_number = 0
    for steps, episodes, train_period, start_goal_reset_period, alpha, free, goal, out_of_b in values:
        run_number += 1
        config["RL Parameters"]["steps"] = str(steps)
        config["RL Parameters"]["episodes"] = str(episodes)
        config["RL Parameters"]["train_period"] = str(train_period)
        config["RL Parameters"]["start_goal_reset_period"] = str(start_goal_reset_period)

        config["SQN Parameters"]["alpha"] = str(alpha)

        config["Gridworld Parameters"]["FREE"] = str(free)
        config["Gridworld Parameters"]["GOAL"] = str(goal)
        config["Gridworld Parameters"]["OUT_OF_BOUNDS"] = str(out_of_b)

        print(f"Run {run_number} on {total_len}")
        main(config)


def bench3():

    grid_size = 10

    runs = [
        "logs/configs/20210714182131.ini",
        "logs/configs/20210714185123.ini",
        "logs/configs/20210714202742.ini"
    ]


    total_len = len(runs)

    run_number = 0
    for run_config in runs:
        run_number += 1

        config = configparser.ConfigParser(allow_no_value=True)
        config.read(run_config)

        # fix a value?
        config["RL Parameters"]["episodes"] = str(10)

        print(f"Run {run_number} on {total_len}")
        main(config)

    print(f"{run_number} runs done on {total_len}")



if __name__ == '__main__':
    # move the past runs tensorboard data to old_runs directory to simplify tensorboard representation
    old_runs = os.listdir("runs")
    for run in old_runs:
        if run.startswith("."):
            continue
        shutil.move(f"runs/{run}", "old_runs")

    bench1()
    # bench2()
    # bench3()
