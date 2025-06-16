from src import *


if __name__ == "__main__":
    # Example1: 2 days plot for each single mouse
    example_exp_2days = BehaviorExperiment(exp_id="joe_example_task_2days", exp_template=TEMPLATE_NULL_2DAYS)

    for example_mice in example_exp_2days.mice:
        save_path = path.join(RESULT_PATH, example_mice.exp_id, example_mice.mice_id)
        licking_raster_plot_go_vs_nogo(example_mice, save_name=path.join(save_path, "heatmap_licking.png"))
        performance_heatmap(example_mice, save_name=path.join(save_path, "heatmap_performance.png"))
        daily_summary(example_mice, save_name=path.join(save_path, "daily_summary.png"))

    # Example2: plot for all mice
    example_exp_7days = BehaviorExperiment(exp_id="joe_example_task_7days", exp_template=TEMPLATE_NULL_7DAYS)

    save_path = path.join(RESULT_PATH, example_exp_7days.exp_id, )
    licking_raster_plot_go_vs_nogo(example_exp_7days.mice, save_name=path.join(save_path, "heatmap_licking.png"))
    performance_heatmap(example_exp_7days.mice, save_name=path.join(save_path, "heatmap_performance.png"))
    daily_summary(example_exp_7days.mice, save_name=path.join(save_path, "daily_summary.png"))

    # Example3: plot for subset list of mice
    example_exp_16days = BehaviorExperiment(exp_id="Ai148_SAT", exp_template=TEMPLATE_SAT_16DAYS)

    list_of_mice = general_filter(example_exp_16days.mice, mice_id=("M023", "M027", "M031", "M032"))
    save_path = path.join(RESULT_PATH, example_exp_16days.exp_id, )
    licking_raster_plot_go_vs_nogo(list_of_mice, save_name=path.join(save_path, "heatmap_licking.png"))
    performance_heatmap(list_of_mice, save_name=path.join(save_path, "heatmap_performance.png"))
    daily_summary(list_of_mice, save_name=path.join(save_path, "daily_summary.png"))

    # Example4: plot for subset list of days
    daily_summary(list_of_mice, save_name=path.join(save_path, "daily_summary_only_three_days.png"),
                  col_days=[SatDay.ACC6, SatDay.SAT1, SatDay.SAT2])


