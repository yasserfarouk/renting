import time
import numpy as np
from typing import Union


class Deadline:
    #TODO: fix infinite deadline, currently it is > 1 year
    def __init__(self, ms: int=2**35, rounds: int = None):
        assert ms or rounds
        if ms and ms <= 0:
            raise ValueError(f"ms must be positive but is {ms}")
        if rounds and rounds <= 2:
            raise ValueError(f"rounds must be at least 3 but is {rounds}")

        self.start_time_ms = time.time() * 1000
        self.ms = ms
        self.rounds = rounds
        self.round = 0

    def get_progress(self) -> Union[float, int]:
        if self.rounds:
            progress = self.round / self.rounds
        else:
            progress = (time.time() * 1000 - self.start_time_ms) / self.ms
        
        # clip progress to [0, 1]
        return min(max(progress, 0), 1)

    def reached(self) -> bool:
        return True if self.get_progress() == 1 else False

    def advance_round(self):
        self.round += 1
