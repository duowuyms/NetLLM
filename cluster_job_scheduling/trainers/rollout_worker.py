import logging
import sys
from abc import ABC, abstractmethod
import os.path as osp
import random

import gymnasium as gym
from gymnasium import logger
from gymnasium.core import ObsType, ActType
import torch

from spark_sched_sim.wrappers import *
from spark_sched_sim.schedulers import make_scheduler
from .utils import Profiler
from spark_sched_sim.metrics import *



class RolloutBuffer:
    def __init__(self, async_rollouts=False):
        self.obsns: list[ObsType] = []
        self.wall_times: list[float] = []
        self.actions: list[ActType] = []
        self.lgprobs: list[float] = []
        self.rewards: list[float] = []
        self.resets = set() if async_rollouts else None

    def add(self, obs, wall_time, action, lgprob, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]
        self.lgprobs += [lgprob]

    def add_reset(self, step):
        assert self.resets is not None, 'resets are for async rollouts only.'
        self.resets.add(step)

    def __len__(self):
        return len(self.obsns)



class RolloutWorker(ABC):
    def __init__(self):
        self.reset_count = 0


    def __call__(
        self,
        rank, 
        conn,
        agent_cls, 
        env_kwargs, 
        agent_kwargs, 
        stdout_dir,
        base_seed,
        seed_step,
        lock
    ):
        self.rank = rank
        self.conn = conn
        self.base_seed = base_seed
        self.seed_step = seed_step
        self.reset_count = 0

        # log each of the processes to separate files
        sys.stdout = open(osp.join(stdout_dir, f'{rank}.out'), 'a')

        self.agent = make_scheduler(agent_kwargs)
        self.agent.actor.eval()

        # might need to download dataset, and only one process should do this.
        # this can be achieved using a lock, such that the first process to 
        # acquire it downloads the dataset, and any subsequent processes notices 
        # that the dataset is already present once it acquires the lock.
        with lock:
            env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_kwargs)
        logger.set_level(logging.INFO)
        env = StochasticTimeLimit(env, env_kwargs['mean_time_limit'])
        env = NeuralActWrapper(env)
        env = self.agent.obs_wrapper_cls(env)
        self.env = env

        # IMPORTANT! Each worker needs to produce unique rollouts, which are 
        # determined by the rng seed
        torch.manual_seed(rank)
        random.seed(rank)

        # torch multiprocessing is very slow without this
        torch.set_num_threads(1)

        self.run()


    def run(self):
        while data := self.conn.recv():
            # load updated model parameters
            self.agent.actor.load_state_dict(data['actor_sd'])
            
            try:
                with Profiler(100): #, HiddenPrints():
                    rollout_buffer = self.collect_rollout()

                self.conn.send({
                    'rollout_buffer': rollout_buffer, 
                    'stats': self.collect_stats()
                })

            except AssertionError as msg:
                print(msg, '\naborting rollout.', flush=True)
                self.conn.send(None)

    
    @abstractmethod
    def collect_rollout(self) -> RolloutBuffer:
        pass


    @property
    def seed(self):
        return self.base_seed + self.seed_step * self.reset_count

    
    def collect_stats(self):
        return {
            'avg_job_duration': self.env.avg_job_duration,
            'avg_num_jobs': avg_num_jobs(self.env),
            'num_completed_jobs': self.env.num_completed_jobs,
            'num_job_arrivals': \
                self.env.num_completed_jobs + self.env.num_active_jobs
        }



class RolloutWorkerSync(RolloutWorker):
    '''model updates are synchronized with environment resets'''

    def collect_rollout(self):
        rollout_buffer = RolloutBuffer()

        obs, _ = self.env.reset(seed=self.seed)
        self.reset_count += 1
        
        wall_time = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, lgprob = self.agent(obs)

            new_obs, reward, terminated, truncated, info = \
                self.env.step(action)
            next_wall_time = info['wall_time']

            rollout_buffer.add(
                obs, wall_time, list(action.values()), lgprob, reward)

            obs = new_obs
            wall_time = next_wall_time

        rollout_buffer.wall_times += [wall_time]

        return rollout_buffer



class RolloutWorkerAsync(RolloutWorker):
    '''model updates occur at regular intervals, regardless of when the 
    environment resets
    '''
    def __init__(self, rollout_duration):
        super().__init__()
        self.rollout_duration = rollout_duration
        self.next_obs = None
        self.next_wall_time = 0.


    def collect_rollout(self):
        rollout_buffer = RolloutBuffer(async_rollouts=True)

        if self.reset_count == 0:
            self.next_obs, _ = self.env.reset(seed=self.seed)
            self.reset_count += 1

        elapsed_time = 0
        step = 0
        while elapsed_time < self.rollout_duration:
            obs, wall_time = self.next_obs, self.next_wall_time

            action, lgprob = self.agent(obs)

            self.next_obs, reward, terminated, truncated, info = \
                self.env.step(action)
            
            self.next_wall_time = info['wall_time']

            rollout_buffer.add(
                obs, elapsed_time, list(action.values()), lgprob, reward)
            
            # add the duration of the this step to the total
            elapsed_time += self.next_wall_time - wall_time

            if terminated or truncated:
                self.next_obs, _ = self.env.reset(seed=self.seed)
                self.reset_count += 1
                self.next_wall_time = 0
                rollout_buffer.add_reset(step)

            step += 1

        rollout_buffer.wall_times += [elapsed_time]

        return rollout_buffer