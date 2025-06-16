from src import *


if __name__ == "__main__":
    example_exp = BehaviorExperiment(exp_id="joe_example_task_2days", exp_template="SAT")

    # plot for each single mouse
    for example_mice in example_exp.mice:
        save_path = path.join(RESULT_PATH, example_mice.exp_id, example_mice.mice_id)
        plot_heatmap_licking(example_mice, save_name=path.join(save_path, "heatmap_licking.png"))
        plot_heatmap_performance(example_mice, save_name=path.join(save_path, "heatmap_performance.png"))
        plot_daily_summary([example_mice, ], save_name=path.join(save_path, "daily_summary.png"))

    # plot for whole experiment



