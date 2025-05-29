import numpy as np
import os
import os.path as path
from collections import defaultdict

from src.ploters.plotting_params import *
from src.config import *


def quick_save(fig, save_name):
    if FIGURE_SHOW_FLAG:
        plt.show()
    else:
        save_path = path.join(RESULT_PATH, save_name)
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, transparent=True)
    plt.close(fig)


def row_col_from_n_subplots(n_subplots: int) -> Tuple[int, int]:
    n_row = int(np.floor(np.sqrt(n_subplots)))
    if n_subplots == n_row * n_row:
        return n_row, n_row
    elif n_subplots <= n_row*(n_row+1):
        return n_row, n_row+1
    else:
        return n_row+1, n_row+1


def bin_average(times: np.ndarray | List, values: np.ndarray | List, bin_width: float, overlap=0.5):
    times = np.array(times)
    values = np.array(values)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]

    step = bin_width * (1 - overlap)
    centers = np.arange(times.min() + bin_width / 2, times.max() - bin_width / 2, step)

    bin_times, bin_avgs, bin_vars = [], [], []
    for c in centers:
        mask = (times >= c - bin_width / 2) & (times <= c + bin_width / 2)
        if np.any(mask):
            bin_times.append(c)
            bin_avgs.append(np.mean(values[mask]))
            bin_vars.append(np.std(values[mask])/len(values[mask]))
    return np.array(bin_times), np.array(bin_avgs), np.array(bin_vars)


def bin_count(times: np.ndarray | List, bin_width: float, overlap=0.5):
    times = np.array(sorted(times))

    step = bin_width * (1 - overlap)
    centers = np.arange(times.min() + bin_width / 2, times.max() - bin_width / 2, step)

    bin_times, bin_avgs = [], []
    for c in centers:
        mask = (times >= c - bin_width / 2) & (times <= c + bin_width / 2)
        bin_times.append(c)
        bin_avgs.append(np.sum(mask))
    return np.array(bin_times), np.array(bin_avgs)

