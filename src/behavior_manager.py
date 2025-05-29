from dataclasses import dataclass, field, MISSING
from typing import List, Callable, Optional, Dict, Any, Iterable, Type
from functools import cached_property
import numpy as np
import os
import os.path as path
from collections import defaultdict
import pandas as pd
from pandas import Timestamp
import glob

from src.utils import *
from src.config import *


@dataclass
class BehaviorTrial:
    exp_id: str
    mice_id: str
    trial_id: int

    lick_times: np.ndarray = field(repr=False)
    exp_start: datetime
    trial_start: Timestamp
    trial_type: BehaviorTrialType

    exp_template: str = field(default=None)

    def __post_init__(self):
        assert self.exp_template in (None, "SAT", "PSE")
        assert self.lick_times.ndim == 1
        self.lick_times = self.lick_times[~np.isnan(self.lick_times)]

    @cached_property
    def day_type(self) -> Type[DayType]:
        if self.exp_template is None:
            return NullDay
        elif self.exp_template == "SAT":
            return SatDay
        elif self.exp_template == "PSE":
            return PseDay
        else:
            raise NotImplementedError

    @cached_property
    def anticipatory_licking(self) -> float:
        return (np.sum((self.lick_times >= ANTICIPATORY_LICKING_RANGE[0]) &
                       (self.lick_times <= ANTICIPATORY_LICKING_RANGE[1])) /
                (ANTICIPATORY_LICKING_RANGE[1] - ANTICIPATORY_LICKING_RANGE[0]))

    @cached_property
    def elapsed_time(self) -> float:
        return (self.trial_start - self.exp_start + time2yesterday(self.exp_start)).total_seconds()

    @cached_property
    def elapsed_hour(self) -> float:
        return (self.trial_start - self.exp_start + time2yesterday(self.exp_start)).total_seconds() / 3600

    @cached_property
    def elapsed_day(self) -> float:
        return (self.trial_start - self.exp_start + time2yesterday(self.exp_start)).total_seconds() / (24 * 3600)

    @cached_property
    def daily_hour(self) -> float:
        return self.elapsed_hour - self.day_idx * 24

    @cached_property
    def daily_day(self) -> float:
        return self.elapsed_day - self.day_idx

    @cached_property
    def num_licks(self) -> int:
        return len(self.lick_times)

    @cached_property
    def day_idx(self) -> int:
        return int(self.elapsed_time / (24 * 3600))

    @cached_property
    def mice_uid(self) -> MiceUID:
        return MiceUID(exp_id=self.exp_id, mice_id=self.mice_id)


def calculate_last_percent_anticipatory_licking(trial_list: List[BehaviorTrial], p: float = 0.2) -> float:
    n_trials_to_extract = int(len(trial_list) * p)
    if n_trials_to_extract <= 0:
        return np.nan
    else:
        sorted_trial_list = list(sorted(trial_list, key=lambda x: x.elapsed_time))
        return np.mean([single_trial.anticipatory_licking
                        for single_trial in sorted_trial_list[-n_trials_to_extract:]])


@dataclass
class BehaviorMice:
    exp_id: str
    mice_id: str
    exp_template: str = field(default=None)

    start_time_exp: datetime = field(init=False)
    raw_df: pd.DataFrame = field(init=False, repr=False)
    data_df: pd.DataFrame = field(init=False, repr=False)
    trials: List[BehaviorTrial] = field(init=False, repr=False)

    def __post_init__(self):
        assert self.exp_template in (None, "SAT", "PSE")
        print(f"\nLoading {self.exp_id} {self.mice_id}")
        self.load_behavior_data()
        self.extract_trials()

    def load_behavior_data(self):
        txt_files = sorted(list(glob.glob(path.join(self.data_path, "*.txt"))))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {self.data_path}")

        txt_files.sort(key=lambda f: parser_start_time_from_filename(f))
        print(f"Found {len(txt_files)} files. Merging...")
        all_data, self.start_time_exp, first_file_start_dt = [], None, None
        for i, file_path in enumerate(txt_files):
            file_start_dt = parser_start_time_from_filename(file_path)
            df_part = pd.read_csv(file_path, header=None, names=TXT_FILE_COL)
            if i == 0:
                self.start_time_exp = file_start_dt if self.mice_id not in MISALIGNED_MICE_RECORDING_START else \
                    MISALIGNED_MICE_RECORDING_START[self.mice_id]
                first_file_start_dt = file_start_dt
            else:
                time_diff = file_start_dt - first_file_start_dt
                df_part['TimeS'] += time_diff.total_seconds()
            df_part['AbsTime'] = first_file_start_dt + pd.to_timedelta(df_part['TimeS'], unit='s')
            all_data.append(df_part)
        self.raw_df = pd.concat(all_data, ignore_index=True)
        print(f"Experiment start time: {self.start_time_exp}")
        pd.set_option('display.max_rows', None)
        if DEBUG_FLAG:
            print(self.raw_df)
        self.preprocess_data()

    def preprocess_data(self):
        df_proc = self.raw_df.copy()

        # calculate time
        df_proc['Phase_prev'] = df_proc['Phase'].shift(1)
        df_proc["IsGo"] = np.where(df_proc['Phase'] == GO_PHASE_CODE, True, np.nan)
        df_proc["IsNoGo"] = np.where(df_proc['Phase'] == NOGO_PHASE_CODE, True, np.nan)

        is_zero = df_proc['RandDelayMS'] == 0
        is_nonzero = df_proc['RandDelayMS'] != 0
        prev_was_zero = is_zero.shift(1).fillna(True)
        prev_was_nonzero = is_nonzero.shift(1).fillna(False)
        # next_is_nonzero = is_nonzero.shift(-1).fillna(False)
        df_proc["TriaStartFlag"] = is_nonzero & prev_was_zero
        df_proc["TriaOnsetFlag"] = is_zero & prev_was_nonzero
        # df_proc["TriaEndFlag"] = is_zero & next_is_nonzero
        df_proc["TrialID"] = df_proc["TriaStartFlag"].cumsum()

        df_proc["RandomDelayS"] = (df_proc["RandDelayMS"].where(df_proc["TriaStartFlag"]).ffill() / 1000.)
        df_proc["TrialOnsetAbsTime"] = df_proc["AbsTime"].where(df_proc["TriaOnsetFlag"])
        df_proc["TrialOnsetAbsTime_fill"] = df_proc.groupby('TrialID')['TrialOnsetAbsTime'].transform('first')

        # df_proc["TrialOnsetS"] = df_proc["TimeS"].where(df_proc["TriaOnsetFlag"])
        df_proc["TrialOnsetS"] = df_proc["TimeS"].where(df_proc["TriaStartFlag"]) + df_proc["RandomDelayS"]
        df_proc["TrialOnsetS_fill"] = df_proc.groupby('TrialID')['TrialOnsetS'].transform('first')
        df_proc['LickTimeS'] = df_proc["TimeS"].where(df_proc['LickStatus'] == LICK_STATUS_CODE)
        df_proc['L0'] = df_proc['LickTimeS'] - df_proc['TrialOnsetS_fill']

        print("Preprocessing complete.")
        self.data_df = df_proc[['TrialOnsetAbsTime_fill', 'TrialID', 'IsGo', 'IsNoGo', 'L0']].copy()
        if DEBUG_FLAG:
            print(self.data_df)

    def extract_trials(self):
        self.trials = []
        for key, group in self.data_df.groupby("TrialID"):
            lick_times = group["L0"].to_numpy()

            assert group["TrialOnsetAbsTime_fill"].nunique() == 1
            assert group["IsGo"].nunique() + group["IsNoGo"].nunique() == 1
            is_go_flag = group["IsGo"].nunique() == 1

            if group['TrialOnsetAbsTime_fill'].unique()[0] < self.start_time_exp:
                continue
            self.trials.append(BehaviorTrial(
                exp_id=self.exp_id,
                mice_id=self.mice_id,
                exp_template=self.exp_template,
                trial_id=int(key),
                lick_times=lick_times,
                exp_start=self.start_time_exp,
                trial_start=group['TrialOnsetAbsTime_fill'].unique()[0],
                trial_type=BehaviorTrialType.Go if is_go_flag else BehaviorTrialType.NoGo
            ))

    @cached_property
    def data_path(self) -> str:
        return path.join(BEHAVIOR_DATA_PATH, self.exp_id, self.mice_id)

    @cached_property
    def mice_uid(self) -> MiceUID:
        return MiceUID(exp_id=self.exp_id, mice_id=self.mice_id)

    @cached_property
    def day_type(self) -> Type[DayType]:
        if self.exp_template is None:
            return NullDay
        elif self.exp_template == "SAT":
            return SatDay
        elif self.exp_template == "PSE":
            return PseDay
        else:
            raise NotImplementedError

    def split_trials_by_days(self) -> Dict[int, List[BehaviorTrial]]:
        trials_by_day = defaultdict(list)
        for single_trial in self.trials:
            trials_by_day[single_trial.day_idx].append(single_trial)
        sorted_day_indices = sorted(list(trials_by_day.keys()))
        return {sorted_day_index: trials_by_day[sorted_day_index] for sorted_day_index in sorted_day_indices}


@dataclass
class BehaviorExperiment:
    exp_id: str
    mice: List[BehaviorMice] = field(init=False, repr=False)
    exp_template: str = field(default=None)

    def __post_init__(self):
        assert self.exp_template in (None, "SAT", "PSE")

        self.mice = []
        for mice_name in os.listdir(self.data_path):
            if path.isfile(path.join(self.data_path, mice_name)):
                continue

            new_mice = BehaviorMice(
                exp_id=self.exp_id,
                exp_template=self.exp_template,
                mice_id=mice_name,
            )
            self.mice.append(new_mice)

    @cached_property
    def day_type(self) -> Type[DayType]:
        if self.exp_template is None:
            return NullDay
        elif self.exp_template == "SAT":
            return SatDay
        elif self.exp_template == "PSE":
            return PseDay
        else:
            raise NotImplementedError

    @cached_property
    def data_path(self) -> str:
        return path.join(BEHAVIOR_DATA_PATH, self.exp_id)
