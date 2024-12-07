import numpy as np

from .task import Task



class Stage:
    def __init__(
        self, 
        id: int, 
        job_id: int, 
        num_tasks: int, 
        task_duration_data: object
    ):
        self.id_ = id
        self.job_id = job_id
        self.task_duration_data = task_duration_data
        self.most_recent_duration = \
            self._rough_task_duration(task_duration_data)
        self.num_tasks = num_tasks
        self.remaining_tasks = set(
            Task(id_=i, stage_id=self.id_, job_id=self.job_id) 
            for i in range(num_tasks)
        )
        self.num_remaining_tasks = num_tasks
        self.num_processing_tasks = 0
        self.num_completed_tasks = 0
        self.is_schedulable = False


    def __hash__(self):
        return hash(self.pool_key)


    def __eq__(self, other):
        if type(other) is type(self):
            return self.pool_key == other.pool_key
        else:
            return False


    @property
    def pool_key(self):
        return (self.job_id, self.id_)


    @property
    def job_pool_key(self):
        return (self.job_id, None)


    @property
    def completed(self):
        return self.num_completed_tasks == self.num_tasks


    @property
    def num_saturated_tasks(self):
        return self.num_processing_tasks + self.num_completed_tasks


    @property
    def next_task_id(self):
        return self.num_saturated_tasks


    @property
    def approx_remaining_work(self):
        return self.most_recent_duration * self.num_remaining_tasks


    def start_on_next_task(self):
        assert self.num_saturated_tasks < self.num_tasks
        task = self.remaining_tasks.pop()
        self.num_remaining_tasks -= 1
        self.num_processing_tasks += 1
        return task


    def add_task_completion(self):
        self.num_processing_tasks -= 1
        self.num_completed_tasks += 1
        

    def _rough_task_duration(self, task_duration_data):
        def durations(key):
            durations = task_duration_data[key].values()
            durations = [i for l in durations for i in l]
            return durations

        all_durations = \
             durations('fresh_durations') + \
             durations('first_wave') + \
             durations('rest_wave')

        return np.mean(all_durations)


    
    
