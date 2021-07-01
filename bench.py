from datetime import datetime

from main import main
import configparser


def bench1():
    steps_s = [100]
    episodes_s = [2000]
    train_period_s = [5]
    start_goal_reset_period_s = [1]

    grid_size = 10

    alpha_s = [2.0]

    # rewards:
    FREE_S = [-0.01]
    GOAL_S = [10.0]
    OUT_OF_BOUNDS_S = [0.0]

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config.ini")
    config["Grid Parameters"]["grid_size"] = str(grid_size)

    total_len = len(steps_s) * len(episodes_s) * len(train_period_s) * len(start_goal_reset_period_s) * len(
        alpha_s) * len(FREE_S) * len(GOAL_S) * len(OUT_OF_BOUNDS_S)

    run_number = 0
    for steps in steps_s:
        for episodes in episodes_s:
            for train_period in train_period_s:
                for start_goal_reset_period in start_goal_reset_period_s:
                    for alpha in alpha_s:
                        for free in FREE_S:
                            for goal in GOAL_S:
                                for out_of_b in OUT_OF_BOUNDS_S:
                                    run_number += 1
                                    config["RL Parameters"]["steps"] = str(steps)
                                    config["RL Parameters"]["episodes"] = str(episodes)
                                    config["RL Parameters"]["train_period"] = str(train_period)
                                    config["RL Parameters"]["start_goal_reset_period"] = str(start_goal_reset_period)

                                    config["SQN Parameters"]["alpha"] = str(alpha)

                                    config["Gridworld Parameters"]["FREE"] = str(free)
                                    config["Gridworld Parameters"]["GOAL"] = str(goal)
                                    config["Gridworld Parameters"]["OUT_OF_BOUNDS"] = str(out_of_b)

                                    now = datetime.now()
                                    with open(f"logs/configs/{now.strftime('%Y%m%d%H%M%S')}.ini", 'w') as configfile:
                                        config.write(configfile)

                                    print(f"Run {run_number} on {total_len}")
                                    main(f"logs/configs/{now.strftime('%Y%m%d%H%M%S')}.ini")


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

        now = datetime.now()
        with open(f"logs/configs/{now.strftime('%Y%m%d%H%M%S')}.ini", 'w') as configfile:
            config.write(configfile)

        print(f"Run {run_number} on {total_len}")
        main(f"logs/configs/{now.strftime('%Y%m%d%H%M%S')}.ini")


if __name__ == '__main__':
    bench1()
    # bench2()
