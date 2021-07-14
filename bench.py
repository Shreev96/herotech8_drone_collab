from datetime import datetime

from main import main
import configparser
import itertools


def bench1():
    hyperparameters = {
        "steps" : [100],
        "episodes" : [2000],
        "train_period" : [5],
        "start_goal_reset_period" : [1],
        "grid_reset_period" : [1],

        "alpha" : [0.5, 1.0, 2.0, 3.0],
        "update_period" : [3, 5, 10, 20],
        "batch_size" : [64, 512],

        # rewards:
        "FREE" : [-0.01],
        "GOAL" : [1.0, 10.0],
        "OUT_OF_BOUNDS" : [-0.01],
        "OBSTACLES" : [-0.1]
    }

    params_values = [v for v in hyperparameters.values()]

    grid_size = 10

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config10.ini")
    config["Grid Parameters"]["grid_size"] = str(grid_size)

    total_len = len(itertools.product(*params_values))

    run_number = 0
    for (steps, episodes, train_period, start_goal_reset_period, grid_reset_period, 
        alpha, update_period, batch_size,
        free, goal, out_of_b, obstacle) in itertools.product(*params_values):

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
        config["Gridworld Parameters"]["OUT_OF_BOUNDS"] = str(out_of_b)
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
        "logs/configs/20210713152528.ini",
        "logs/configs/20210713152731.ini"
    ]

    config = configparser.ConfigParser(allow_no_value=True)

    total_len = len(runs)

    run_number = 0
    for run_config in runs:
        run_number += 1

        config.read(run_config)

        # fix a value?
        config["RL Parameters"]["episodes"] = str(10000)
        config["SQN Parameters"]["batch_size"] = str(512)

        now = datetime.now()
        with open(f"runs/{now.strftime('%Y%m%d%H%M%S')}.ini", 'w') as configfile:
            config.write(configfile)

        print(f"Run {run_number} on {total_len}")
        main(config)

    print(f"{run_number} runs done on {total_len}")



if __name__ == '__main__':
    bench1()
    # bench2()
    # bench3()
