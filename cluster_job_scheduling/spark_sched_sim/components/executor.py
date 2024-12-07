from typing import Optional, List

from .task import Task


class Executor:
    
    def __init__(self, id_: int):
        # index of this operation within its operation
        self.id_ = id_

        # task that this executor is or just finished executing,
        # or `None` if the executor is idle
        self.task: Optional[Task] = None

        # id of current job that this executor is local to, if any
        self.job_id: Optional[int] = None

        # whether or not this executing is executing a task.
        # NOTE: can be `False` while `self.task is not None`,
        # if the executor just finished executing
        self.is_executing = False

        # list of pairs [t, job_id], where `t` is the wall time that this executor
        # was released from job with id `job_id`, or `None` if it has not been released
        # yet. `job_id` is -1 if the executor is at the general pool.
        # NOTE: only used for rendering
        self.history: List[list] = [[None, -1]]


    
    @property
    def is_idle(self):
        return self.task is None
    


    def is_at_job(self, job_id):
        return self.job_id == job_id



    def add_history(self, wall_time, job_id):
        '''should be called whenever this executor is released from a job'''
        if self.history is None:
            self.history = []

        if len(self.history) > 0:
            # add release time to most recent history
            self.history[-1][0] = wall_time
        
        # add new history
        self.history += [[None, job_id]]

    

