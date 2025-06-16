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


# For misalignment, some files doesn't correct start time in the file name,
# you can manually set the correct experiment start time here
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

# time window take into account, relative to trial onset
BEHAVIOR_RANGE = (-1, 3.5)  # s
ANTICIPATORY_LICKING_RANGE = (0.7, 1)  # s

