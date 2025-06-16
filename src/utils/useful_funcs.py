import re
from datetime import datetime, timedelta
from typing import Optional, List, Iterable


def parser_start_time_from_filename(filename: str) -> datetime:
    match = re.search(r"(\d{2})_(\d{2})_(\d{2})_T_(\d{2})_(\d{2})_(\d{2})", filename)
    if match:
        month, day, year, hour, minute, second = map(int, match.groups())
        return datetime(2000 + year, month, day, hour, minute, second)
    else:
        raise ValueError(f"Could not parse start time from filename: {filename}")


def time2tomorrow(now_time: datetime):
    tomorrow = now_time + timedelta(days=1)
    midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
    time_to_midnight = midnight - now_time
    return time_to_midnight


def time2yesterday(now_time: datetime):
    midnight = datetime(now_time.year, now_time.month, now_time.day, 0, 0, 0)
    time_to_midnight = now_time - midnight
    return time_to_midnight


def time2nearest_noon(now_time: datetime):
    nearest_noon = datetime(now_time.year, now_time.month, now_time.day, 12, 0, 0)
    time_to_nearest_noon = now_time - nearest_noon
    return time_to_nearest_noon


def general_filter(datalist: Iterable, **criteria) -> list:
    def matches(one_data) -> bool:
        def check_criterion(key, value):
            retrieved_val = getattr(one_data, key, None)
            if callable(value):
                return value(retrieved_val)
            elif isinstance(value, tuple):
                return retrieved_val in value
            else:
                return retrieved_val == value
        return all(check_criterion(k, v) for k, v in criteria.items())
    return [d for d in datalist if matches(d)]


