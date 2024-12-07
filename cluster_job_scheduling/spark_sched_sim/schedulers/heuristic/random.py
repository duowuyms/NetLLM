import numpy as np

from .heuristic import HeuristicScheduler



class RandomScheduler(HeuristicScheduler):

    def __init__(self, seed=42):
        super().__init__('Random')
        self.set_seed(seed)

    
    def set_seed(self, seed):
        self.np_random = np.random.RandomState(seed)
    

    def schedule(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.exec_supplies)

        job_idxs = list(range(num_active_jobs))
        stage_idx = -1
        while len(job_idxs) > 0:
            j = self.np_random.choice(job_idxs)
            stage_idx = self.find_stage(obs, j)
            if stage_idx != -1:
                break
            else:
                job_idxs.remove(j)

        num_exec = self.np_random.randint(1, obs.num_committable_execs + 1)

        return {
            'stage_idx': stage_idx,
            'num_exec': num_exec
        }
        