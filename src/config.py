import numpy as np
import os
import os.path as path
from datetime import datetime

from src.utils.terminology import *

# average policy
DEBUG_FLAG = False
FIGURE_SHOW_FLAG = False


# path to data
ROOT_PATH = r"C:\Users\maxyc\PycharmProjects\behavior-mini"
BEHAVIOR_DATA_PATH = path.join(ROOT_PATH, "data")
RESULT_PATH = path.join(ROOT_PATH, "result")


# For misalignment, some files doesn't correct start time in the file name
MISALIGNED_MICE_RECORDING_START = {
    MiceUID(exp_id="Ai148_SAT", mice_id="M031"):
        datetime(2021, 11, 9, 12, 0, 0)
}


# behavior data reading
TXT_FILE_COL = ('TimeS', 'PokeStatus', 'LickStatus', 'Phase', 'RandDelayMS')
TRIAL_RANGE = (-2, 4)  # s

# Go NoGo
GO_PHASE_CODE = 3
NOGO_PHASE_CODE = 9
LICK_STATUS_CODE = 2

# plotting config
D_TRIAL_SMALL = 0.01  # s
D_DAY_SMALL = 0.02  # day
D_TRIAL_LARGE = 0.1  # s
D_DAY_LARGE = 0.2  # day
BEHAVIOR_RANGE = (-1, 3.5)  # s
DAY_TEXT_SIZE = 5
ANTICIPATORY_LICKING_RANGE = (0.7, 1)

# plotting related
BEHAVIOR_BIN_SIZE_DAY = 0.1
BEHAVIOR_BIN_SIZE_HOUR = 2
BEHAVIOR_BIN_SIZE_TRIAL = 0.1
