from gymnasium import Wrapper
import numpy as np



class StochasticTimeLimit(Wrapper):
    '''Samples each episode's time limit from an exponential distribution'''

    def __init__(self, env, mean_time_limit, seed=42):
        super().__init__(env)
        self.mean_time_limit = mean_time_limit
        self.np_random = np.random.RandomState(seed)


    def reset(self, seed=None, options=None):
        '''samples a new time limit prior to resetting'''
        if seed:
            self.np_random = np.random.RandomState(seed)
        self.time_limit = self.np_random.exponential(self.mean_time_limit)
        print(f'resetting. seed={seed}, timelim={int(self.time_limit*1e-3)}s', 
              flush=True)
        if not options:
            options = {}
        options['time_limit'] = self.time_limit
        return self.env.reset(seed=seed, options=options)
    

    def step(self, act):
        '''modifies `truncated` signal when time limit is reached'''
        obs, rew, term, trunc, info = self.env.step(act)
        if info['wall_time'] >= self.time_limit:
            trunc = True
        return obs, rew, term, trunc, info

