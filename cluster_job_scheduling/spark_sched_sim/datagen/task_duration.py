import numpy as np
import random



class TaskDurationGen:
    def __init__(self, exec_cap, warmup_delay):
        self.warmup_delay = warmup_delay
        self._init_executor_intervals(exec_cap)
        self.np_random = None


    def reset(self, np_random):
        self.np_random = np_random


    def sample(
        self, 
        task, 
        executor, 
        num_local_executors,
        task_duration_data
    ):
        assert num_local_executors > 0
        assert self.np_random

        self.task_duration_data = task_duration_data

        # sample an executor point in the data
        executor_key = self._sample_executor_key(num_local_executors)

        if executor.is_idle:
            # the executor was just sitting idly or moving between jobs, so it needs time to warm up
            try:
                return self._sample('fresh_durations', executor_key)
            except:
                return self._sample('first_wave', executor_key, warmup=True)
        

        if executor.task.stage_id == task.stage_id:
            # the executor is continuing work on the same stage, which is relatively fast
            try:
                return self._sample('rest_wave', executor_key)
            except:
                pass

        # the executor is new to this stage (or 'rest_wave' data was not available)
        try:
            return self._sample('first_wave', executor_key)
        except:
            return self._sample('fresh_durations', executor_key)


    def _sample(self, wave, executor_key, warmup=False):
        '''raises an exception if `executor_key` is not found in the durations from `wave`'''
        durations = self.task_duration_data[wave][executor_key]
        duration = random.choice(durations)
        if warmup:
            duration += self.warmup_delay
        return duration


    def _sample_executor_key(self, num_local_executors):
        left_exec, right_exec = self.executor_intervals[num_local_executors]

        executor_key = None

        if left_exec == right_exec:
            executor_key = left_exec
        else:
            # faster than random.randint
            rand_pt = 1 + int(random.random() * (right_exec - left_exec))
            if rand_pt <= num_local_executors - left_exec:
                executor_key = left_exec
            else:
                executor_key = right_exec

        if executor_key not in self.task_duration_data['first_wave']:
            # more executors than number of tasks in the job
            executor_key = max(self.task_duration_data['first_wave'])

        return executor_key
    

    def _init_executor_intervals(self, exec_cap):
        exec_levels = [5, 10, 20, 40, 50, 60, 80, 100]

        intervals = np.zeros((exec_cap+1, 2))

        # get the left most map
        intervals[:exec_levels[0]+1] = exec_levels[0]

        # get the center map
        for i in range(len(exec_levels) - 1):
            intervals[exec_levels[i]+1 : exec_levels[i+1]] = (exec_levels[i], exec_levels[i+1])
            
            if exec_levels[i+1] > exec_cap:
                break

            # at the data point
            intervals[exec_levels[i+1]] = exec_levels[i+1]

        # get the residual map
        if exec_cap > exec_levels[-1]:
            intervals[exec_levels[-1]+1 : exec_cap] = exec_levels[-1]

        self.executor_intervals = intervals