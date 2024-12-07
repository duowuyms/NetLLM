import heapq
import itertools
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class TimelineEvent:
    class Type(Enum):
        JOB_ARRIVAL = auto()
        TASK_COMPLETION = auto()
        EXECUTOR_ARRIVAL = auto()

    type: Type

    data: dict



# heap-based timeline

class Timeline:
    def __init__(self):
        # priority queue
        self._pq = []

        # tie breaker
        self._counter = itertools.count()


    def __len__(self):
        return len(self._pq)
    

    @property
    def empty(self):
        return len(self) == 0
    

    def peek(self):
        try:
            key, _, item = self._pq[0]
            return key, item
        except:
            return None, None


    def push(self, key, item):
        heapq.heappush(self._pq, (key, next(self._counter), item))
        

    def pop(self):
        if len(self._pq) > 0:
            key, _, item = heapq.heappop(self._pq)
            return key, item
        else:
            return None, None
        

    def reset(self):
        self._pq = []
        self._counter = itertools.count()

    
    def events(self):
        return (event for (*_, event) in self._pq)