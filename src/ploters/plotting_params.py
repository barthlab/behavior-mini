import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import splprep, splev
import matplotlib.patches as mpatches

from src.behavior_manager import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 5
plt.rcParams["figure.dpi"] = 300

plt.rcParams.update({
    'xtick.labelsize': 5,  # X-axis tick labels
    'ytick.labelsize': 5,  # Y-axis tick labels
    'axes.labelsize': 6,  # X and Y axis labels
    'axes.titlesize': 7,  # Plot title
    'legend.fontsize': 7,  # Legend font size
    'figure.titlesize': 8  # Figure title (suptitle)
})

GENERAL_COLORS = {
    "water": "lightblue",
    "nowater": "gray",
    "puff": "gray",
    "annotate": 'gray',
    "SAT": "lightskyblue",
    "PSE": 'lightpink'
}

# lick cmap
GRAY3 = ["#000000", "#969696", "#ffffff"]
BINARY = ['white', 'black']
BEHAVIOR_TRIAL_TYPE2COLOR = {
    BehaviorTrialType.Go: "green",
    BehaviorTrialType.NoGo: "red",
}

PERFORMANCE_CMAP = LinearSegmentedColormap(
    'red_to_green_cmap',
    {
        'red': [(0.0, 1.0, 1.0),  # Red at the start
                (0.5, 1.0, 1.0),  # White in the middle (Red component is 1)
                (1.0, 0.0, 0.0)],  # No red at the end (Green)

        'green': [(0.0, 0.0, 0.0),  # No green at the start (Red)
                  (0.5, 1.0, 1.0),  # White in the middle (Green component is 1)
                  (1.0, 1.0, 1.0)],  # Green at the end

        'blue': [(0.0, 0.0, 0.0),  # No blue at the start (Red)
                 (0.5, 1.0, 1.0),  # White in the middle (Blue component is 1)
                 (1.0, 0.0, 0.0)]  # No blue at the end (Green)
    },
    N=256)


# Plotting Parameters Config

# bin size related params
BIN_TRIAL_RASTER_LICKING = 0.01  # s
BIN_DAY_RASTER_LICKING = 0.02  # day
ASPECT_RASTER_LICKING = BIN_DAY_RASTER_LICKING / (BIN_TRIAL_RASTER_LICKING * 1.5)

BIN_TRIAL_HEATMAP_PERFORMANCE = 0.1  # s
BIN_DAY_HEATMAP_PERFORMANCE = 0.1  # day
ASPECT_HEATMAP_PERFORMANCE = BIN_DAY_HEATMAP_PERFORMANCE / (BIN_TRIAL_HEATMAP_PERFORMANCE * 1.5)

# BIN_DAY_DAILY_SUMMARY = 0.1
BIN_HOUR_DAILY_SUMMARY = 2  # hr
BIN_TRIAL_DAILY_SUMMARY = 0.1  # s

