from enum import Enum
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any, Iterable, Type, Tuple, Union


# trial type enum
class BehaviorTrialType(Enum):
    Go = 1
    NoGo = 0


# day enum
class NullDay(Enum):
    Day1 = 0
    Day2 = 1
    Day3 = 2
    Day4 = 3
    Day5 = 4
    Day6 = 5
    Day7 = 6
    Day8 = 7
    Day9 = 8
    Day10 = 9
    Day11 = 10
    Day12 = 11
    Day13 = 12
    Day14 = 13
    Day15 = 14
    Day16 = 15


class SatDay(Enum):
    ACC1 = 0
    ACC2 = 1
    ACC3 = 2
    ACC4 = 3
    ACC5 = 4
    ACC6 = 5
    SAT1 = 6
    SAT2 = 7
    SAT3 = 8
    SAT4 = 9
    SAT5 = 10
    SAT6 = 11
    SAT7 = 12
    SAT8 = 13
    SAT9 = 14
    SAT10 = 15


class PseDay(Enum):
    ACC1 = 0
    ACC2 = 1
    ACC3 = 2
    ACC4 = 3
    ACC5 = 4
    ACC6 = 5
    PSE1 = 6
    PSE2 = 7
    PSE3 = 8
    PSE4 = 9
    PSE5 = 10
    PSE6 = 11
    PSE7 = 12
    PSE8 = 13
    PSE9 = 14
    PSE10 = 15


DayType = Union[PseDay, SatDay, NullDay]


ADV_SAT: Dict[str, Tuple[DayType, ...]] = {
    "ACC123": tuple(SatDay(i) for i in range(3)),
    "ACC456": tuple(SatDay(i) for i in range(3, 6)),
    "SAT123": tuple(SatDay(i) for i in range(6, 9)),
    "SAT456": tuple(SatDay(i) for i in range(9, 12)),
    "SAT789": tuple(SatDay(i) for i in range(12, 15)),
    "SAT12": tuple(SatDay(i) for i in range(6, 8)),
    "SAT45": tuple(SatDay(i) for i in range(9, 11)),
    "SAT56": tuple(SatDay(i) for i in range(10, 12)),
    "SAT89": tuple(SatDay(i) for i in range(13, 15)),
    "SAT910": tuple(SatDay(i) for i in range(14, 16)),
    "SAT8910": tuple(SatDay(i) for i in range(13, 16)),
}
ADV_SAT.update({SatDay(i).name: tuple((SatDay(i),)) for i in range(16)})

ADV_PSE: Dict[str, Tuple[DayType, ...]] = {
    "ACC123": tuple(PseDay(i) for i in range(3)),
    "ACC456": tuple(PseDay(i) for i in range(3, 6)),
    "PSE123": tuple(PseDay(i) for i in range(6, 9)),
    "PSE456": tuple(PseDay(i) for i in range(9, 12)),
    "PSE789": tuple(PseDay(i) for i in range(12, 15)),
    "PSE12": tuple(PseDay(i) for i in range(6, 8)),
    "PSE23": tuple(PseDay(i) for i in range(7, 9)),
    "PSE45": tuple(PseDay(i) for i in range(9, 11)),
    "PSE56": tuple(PseDay(i) for i in range(10, 12)),
    "PSE89": tuple(PseDay(i) for i in range(13, 15)),
    "PSE910": tuple(PseDay(i) for i in range(14, 16)),
    "PSE8910": tuple(PseDay(i) for i in range(13, 16)),
}
ADV_PSE.update({PseDay(i).name: tuple((PseDay(i),)) for i in range(16)})


@dataclass(frozen=True, order=True)
class MiceUID:
    exp_id: str
    mice_id: str

    def in_short(self) -> str:
        return f"{self.mice_id}"



