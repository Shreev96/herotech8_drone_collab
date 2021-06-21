from datetime import datetime

from main import main
import configparser

if __name__ == '__main__':
    steps_s = [10, 20, 30, 50, 100, 500]
    episodes_s = [1000]
    train_period_s = [1, 3, 10, 50, 100]
    start_goal_reset_period_s = [1, 3, 10, 50, 100]

    grid_size = 10

    alpha_s = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    # rewards:
    FREE_S = [0, -0.04, -0.1, -1.0, -5.0]
    GOAL_S = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    OUT_OF_BOUNDS_S = [-0.8, -1.0, -5.0, -10.0, -100.0]

    config = configparser.ConfigParser(allow_no_value=True)
    config.read("config.ini")

    total_len = len(steps_s) + len(episodes_s) + len(train_period_s) + len(start_goal_reset_period_s) + len(alpha_s) + len(FREE_S) + len(GOAL_S) + len(OUT_OF_BOUNDS_S)

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
