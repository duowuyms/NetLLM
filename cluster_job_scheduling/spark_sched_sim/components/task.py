from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    id_: int
    stage_id: int
    job_id: int
    executor_id: int = None
    t_accepted: float = np.inf
    t_completed: float = np.inf

    @property
    def __unique_id(self):
        return (self.job_id, self.stage_id, self.id_)

    def __hash__(self):
        return hash(self.__unique_id)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__unique_id == other.__unique_id
        else:
            return False