import os
import os.path as path
from collections import defaultdict
from itertools import chain
import datashader as ds
import datashader.transfer_functions as tf
import xarray as xr


from src.behavior_manager import *
from src.ploters.plotting_params import *
from src.ploters.plotting_utils import *


def plot_heatmap_licking(single_mice: BehaviorMice, save_name: str):
    trials_data = []
    for single_trial in single_mice.trials:
        for lick_time in single_trial.lick_times:
            if BEHAVIOR_RANGE[0] <= lick_time <= BEHAVIOR_RANGE[1]:
                trials_data.append({
                    "lick_time": lick_time,
                    'elapsed_day': single_trial.elapsed_day,
                    'trial_type': 'Go' if single_trial.trial_type is BehaviorTrialType.Go else 'NoGo'
                })
    df = pd.DataFrame(trials_data)

    x_range = (df['lick_time'].min(), df['lick_time'].max())
    y_range = (df['elapsed_day'].min(), df['elapsed_day'].max())

    cvs = ds.Canvas(plot_width=int((x_range[1]-x_range[0])/D_TRIAL_SMALL),
                    plot_height=int((y_range[1]-y_range[0])/D_DAY_SMALL),
                    x_range=x_range, y_range=y_range)
    agg_go = cvs.points(df[df['trial_type'] == 'Go'], 'lick_time', 'elapsed_day',
                        agg=ds.count()).astype(float)
    agg_nogo = cvs.points(df[df['trial_type'] == 'NoGo'], 'lick_time', 'elapsed_day',
                          agg=ds.count()).astype(float)
    img_go = tf.shade((agg_go > 0).astype(int), cmap=BINARY)
    img_nogo = tf.shade((agg_nogo > 0).astype(int), cmap=BINARY)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    fig, axs = plt.subplots(1, 2,)

    axs[0].imshow(img_go.to_pil(), extent=extent, origin='upper', aspect='auto', interpolation='nearest')
    axs[1].imshow(img_nogo.to_pil(), extent=extent, origin='upper', aspect='auto', interpolation='nearest')

    for ax_id in range(2):
        for i, day_id in enumerate(single_mice.day_type):
            axs[ax_id].text(-0.05, i + 1, day_id.name, transform=axs[0].get_yaxis_transform(), fontsize=DAY_TEXT_SIZE,
                            ha='right', va='center', clip_on=False)
        axs[ax_id].set_yticks([0.5 + i for i in range(17)], ["" for _ in range(17)])
        axs[ax_id].set_xlabel('Lick Time (s)')
        # axs[ax_id].set_ylim(17, 0)
        axs[ax_id].set_xlim(*BEHAVIOR_RANGE)
        axs[ax_id].set_xticks([0, 1, ])
        axs[ax_id].spines[['right', 'top']].set_visible(False)

    axs[0].axvspan(0, 0.5, lw=0, color=GENERAL_COLORS['puff'], alpha=0.1, zorder=3)
    axs[0].axvline(x=1, lw=1, ls='--', color=GENERAL_COLORS["water"], alpha=0.8)

    axs[0].set_title('Go Trials')
    axs[1].set_title('NoGo Trials')
    fig.suptitle(single_mice.mice_id)

    fig.set_size_inches(3, 5)
    quick_save(fig, save_name=save_name)


def plot_heatmap_performance(single_mice: BehaviorMice, save_name: str):
    trials_data = []
    for single_trial in single_mice.trials:
        for lick_time in single_trial.lick_times:
            if BEHAVIOR_RANGE[0] <= lick_time <= BEHAVIOR_RANGE[1]:
                trials_data.append({
                    "lick_time": lick_time,
                    'elapsed_day': single_trial.elapsed_day,
                    'trial_type': 'Go' if single_trial.trial_type is BehaviorTrialType.Go else 'NoGo',
                })
    df = pd.DataFrame(trials_data)

    unique_trials_data = []
    for i, trial in enumerate(single_mice.trials):
        unique_trials_data.append({
            'trial_id': i,
            'elapsed_day': trial.elapsed_day,
            'trial_type': 'Go' if trial.trial_type is BehaviorTrialType.Go else 'NoGo',
            'dummy_x': 0
        })
    df_trials = pd.DataFrame(unique_trials_data)

    x_range = (df['lick_time'].min(), df['lick_time'].max())
    y_range = (df['elapsed_day'].min(), df['elapsed_day'].max())

    canvas_1d_day = ds.Canvas(plot_width=1, plot_height=int((y_range[1]-y_range[0])/D_DAY_LARGE),
                              x_range=(-0.5, 0.5), y_range=y_range)
    go_trial_counts_2d = canvas_1d_day.points(df_trials[df_trials['trial_type'] == 'Go'],
                                              'dummy_x', 'elapsed_day', ds.count())
    nogo_trial_counts_2d = canvas_1d_day.points(df_trials[df_trials['trial_type'] == 'NoGo'],
                                                'dummy_x', 'elapsed_day', ds.count())
    go_trial_norm_factors = go_trial_counts_2d.isel({"dummy_x": 0})
    nogo_trial_norm_factors = nogo_trial_counts_2d.isel({"dummy_x": 0})

    cvs = ds.Canvas(plot_width=int((x_range[1]-x_range[0])/D_TRIAL_LARGE),
                    plot_height=int((y_range[1]-y_range[0])/D_DAY_LARGE),
                    x_range=x_range, y_range=y_range)
    agg_go = cvs.points(df[df['trial_type'] == 'Go'], 'lick_time', 'elapsed_day',
                        agg=ds.count()).astype(float)
    agg_nogo = cvs.points(df[df['trial_type'] == 'NoGo'], 'lick_time', 'elapsed_day',
                          agg=ds.count()).astype(float)
    normalized_agg_go = xr.where(go_trial_norm_factors > 0, agg_go / go_trial_norm_factors, 0.0)
    normalized_agg_nogo = xr.where(nogo_trial_norm_factors > 0, agg_nogo / nogo_trial_norm_factors, 0.0)

    performance = (normalized_agg_go - normalized_agg_nogo)/(normalized_agg_go + normalized_agg_nogo + D_TRIAL_LARGE)
    img_performance_shaded = tf.shade(performance, cmap=PERFORMANCE_CMAP, how='linear', span=(-1, 1))

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.15], 'wspace': 0.3})

    axs[0].imshow(img_performance_shaded.to_pil(), extent=extent, origin='upper', aspect='auto',
                  interpolation='nearest')

    for i, day_id in enumerate(single_mice.day_type):
        axs[0].text(-0.05, i + 1, day_id.name, transform=axs[0].get_yaxis_transform(), fontsize=DAY_TEXT_SIZE,
                    ha='right', va='center', clip_on=False)
    axs[0].set_yticks([0.5 + i for i in range(17)], ["" for _ in range(17)])
    axs[0].set_xlabel('Lick Time (s)')
    axs[0].set_ylim(17, 0)
    axs[0].set_xlim(-1., 2)
    axs[0].set_xticks([0, 1, ])
    axs[0].spines[['right', 'top']].set_visible(False)

    axs[0].axvline(x=0, lw=1, ls='--', color='black', alpha=0.6)
    axs[0].axvline(x=1, lw=1, ls='--', color='black', alpha=0.6)

    axs[0].set_title('Performance')
    fig.suptitle(single_mice.mice_id)

    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=PERFORMANCE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axs[1], orientation='vertical')
    cbar.set_label('Performance Index (Go - NoGo)/(Go + NoGo)')

    fig.set_size_inches(2, 5)
    quick_save(fig, save_name=save_name)


def plot_single_day_performance(ax: matplotlib.pyplot.Axes, daily_trials: List[BehaviorTrial],):

    for trial_type, scale_factor in zip(
            [BehaviorTrialType.Go, BehaviorTrialType.NoGo],
            [1, -1],
    ):
        daily_specific_trials = general_filter(daily_trials, trial_type=trial_type)
        if len(daily_specific_trials) == 0:
            continue
        trial_times = [single_trial.daily_hour for single_trial in daily_specific_trials]
        trial_lick_freq = [scale_factor * single_trial.anticipatory_licking for single_trial in daily_specific_trials]
        bin_time, bin_lick_freq, bin_var = bin_average(trial_times, trial_lick_freq,
                                                       bin_width=BEHAVIOR_BIN_SIZE_HOUR)

        ax.bar(bin_time, bin_lick_freq, width=BEHAVIOR_BIN_SIZE_HOUR * 0.25, alpha=0.8,
               yerr=bin_var, capsize=1, error_kw={"capthick": 0.4, "elinewidth": 0.4,},
               color=BEHAVIOR_TRIAL_TYPE2COLOR[trial_type])
    ax.plot([0, 24], [0, 0], lw=1, color='gray', alpha=0.8)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both')


def plot_single_day_licking_raster(ax: matplotlib.pyplot.Axes, daily_trials: List[BehaviorTrial], color: str):
    n_trials = len(daily_trials)
    lick_times = []
    for single_trial in daily_trials:
        lick_times += list(single_trial.lick_times)
    if len(lick_times) > 0:
        bin_time, bin_lick_freq = bin_count(lick_times, bin_width=BEHAVIOR_BIN_SIZE_TRIAL)
        bin_lick_freq = bin_lick_freq/(n_trials*BEHAVIOR_BIN_SIZE_TRIAL)
        ax.plot(bin_time, bin_lick_freq, lw=1, alpha=0.8, color=color)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both')


def plot_daily_performance(single_mice: BehaviorMice, save_name: str):
    trials_by_days = single_mice.split_trials_by_days()
    n_day = len(trials_by_days.keys())
    n_row, n_col = row_col_from_n_subplots(n_day)

    fig, axs = plt.subplots(n_row, n_col, sharey='all', sharex='all')

    for day_id, day_name in enumerate(trials_by_days.keys()):
        tmp_ax = axs[int(np.floor(day_id/n_col)), day_id % n_col]
        daily_trials = trials_by_days[day_name]
        plot_single_day_performance(tmp_ax, daily_trials)
        tmp_ax.set_xlabel(f"Day{day_name} (Hour)")

    fig.set_size_inches(2*n_col, 2*n_row)
    fig.tight_layout()
    # quick_save(fig, save_name)
    plt.show()


def plot_daily_summary(
        beh_mice_list: List[BehaviorMice], save_name: str, col_days: List[DayType] = None,
):
    col_days = list(beh_mice_list[0].day_type)
    n_col = len(col_days)

    fig, axs = plt.subplots(2, n_col, sharey='row', height_ratios=[0.8, 1])

    for col_id in range(n_col):
        daily_trials = list(chain.from_iterable([beh_mice.split_trials_by_days().get(
            col_days[col_id].value+1, []) for beh_mice in beh_mice_list]))
        plot_single_day_performance(axs[1, col_id], daily_trials)
        if col_id == 0:
            axs[1, col_id].set_ylabel(f"Anticipatory Licking (Hz)")
        axs[1, col_id].set_xlabel(f"Hour at day")
        axs[1, col_id].set_xticks([0, 12, 24])

        plot_single_day_licking_raster(axs[0, col_id],
                                       general_filter(daily_trials, trial_type=BehaviorTrialType.Go),
                                       color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.Go])
        plot_single_day_licking_raster(axs[0, col_id],
                                       general_filter(daily_trials, trial_type=BehaviorTrialType.NoGo),
                                       color=BEHAVIOR_TRIAL_TYPE2COLOR[BehaviorTrialType.NoGo])
        if col_id == 0:
            axs[0, col_id].set_ylabel(f"Licking Freq (Hz)")
        axs[0, col_id].set_title(col_days[col_id].name)
        axs[0, col_id].set_xlim(*BEHAVIOR_RANGE)
        axs[0, col_id].axvspan(0, 0.5, lw=0, color=GENERAL_COLORS['puff'], alpha=0.4)
        axs[0, col_id].axvline(x=1, lw=1, color=GENERAL_COLORS['water'], alpha=0.7, ls='--')

    fig.set_size_inches(0.85*n_col, 1*2)
    fig.tight_layout()
    quick_save(fig, save_name)
