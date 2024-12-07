from typing import Optional
from bisect import bisect_left, bisect_right
from collections import deque

import numpy as np
from gymnasium import Env
from gymnasium.spaces import *

from .components.executor_tracker import (
    ExecutorTracker,
    COMMON_POOL_KEY
)
from .components.timeline import TimelineEvent
from .components import Executor
from .datagen.tpch_job_sequence import TPCHJobSequenceGen
from .datagen.task_duration import TaskDurationGen
from . import graph_utils
from . import metrics
try:
    from .components.renderer import Renderer
    PYGAME_AVAILABLE = True
except:
    PYGAME_AVAILABLE = False



NUM_NODE_FEATURES = 3
RENDER_FPS = 30


class SparkSchedSimEnv(Env):
    '''Gymnasium environment that simulates job dag scheduling'''

    metadata = {'render_modes': ['human'], 'render_fps': RENDER_FPS}

    def __init__(
        self,
        num_executors: int,
        job_arrival_rate: float,
        moving_delay: float,
        warmup_delay: float,
        dataset: str,
        beta: float = 0,
        job_arrival_cap: Optional[int] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        '''
        Args:
            num_executors (int): number of simulated executors. More executors
                means a higher possible level of parallelism.
            job_arrival_rate (float): non-negative number that controls how
                quickly new jobs arrive into the system. This is the parameter
                of an exponential distributions, and so its inverse is the
                mean job inter-arrival time in ms.
            moving_delay (float): time in ms it takes for a executor to move 
                between jobs
            warmup_delay (float): an executor is slower on its first task from 
                a stage if it was previously idle or moving jobs, which is
                caputred by adding a warmup delay to the task duration
            dataset (str): choice of dataset to generate jobs from. Currently,
                only 'tpch' is supported.
            beta (float): continuous discount factor in [0,+inf). If set to 0,
                then rewards are not discounted.
            job_arrival_cap: (optional int): limit on the number of jobs that 
                arrive throughout the simulation. If set to `None`, then the
                episode ends when a time limit is reached.
            render_mode (optional str): if set to 'human', then a visualization
                of the simulation is rendred in real time
        '''
        self.wall_time = 0
        self.num_executors = num_executors
        self.moving_delay = moving_delay
        self.job_arrival_cap = job_arrival_cap
        self.beta = beta

        # must be ordered
        self.active_job_ids = []
        self.completed_job_ids = set()

        if dataset == 'tpch':
            self.datagen = TPCHJobSequenceGen(
                job_arrival_rate, job_arrival_cap)
        else:
            raise ValueError('unsupported dataset.')

        self.task_duration_gen = TaskDurationGen(
            self.num_executors, warmup_delay)

        self.exec_tracker = ExecutorTracker(num_executors)

        self.handle_event = {
            TimelineEvent.Type.JOB_ARRIVAL: self._handle_job_arrival,
            TimelineEvent.Type.EXECUTOR_ARRIVAL: self._handle_executor_arrival,
            TimelineEvent.Type.TASK_COMPLETION: self._handle_task_completion
        }

        self.render_mode = render_mode
        if render_mode == 'human':
            assert PYGAME_AVAILABLE, 'pygame is unavailable'
            self.renderer = Renderer(
                self.num_executors, 
                job_arrival_cap, 
                render_fps=self.metadata['render_fps']
            )
        else:
            self.renderer = None

        self.job_duration_buff = deque(maxlen=200)

        self.action_space = Dict({
            # stage selection
            # NOTE: upper bound of this space is dynamic, equal to 
            # the number of active stages. Initialized to 1.
            'stage_idx': Discrete(1, start=-1),

            # parallelism limit selection
            'num_exec': Discrete(num_executors, start=1)
        })

        self.observation_space = Dict({
            # shape: (num active stages) x (num node features)
            # stage features: num remaining tasks, most recent task duration, is stage schedulable
            # edge features: none
            # >>> comments of wuduo <<<
            # a stage is a node in a DAG, and a job is a DAG.
            # while the intially features of a stage (node) are of three elements descripbed above,
            # they are transformed later in the codes of five elements:
            # 1. how many exec can be added to each node; 2.whether or not a node belongs to the source job,
            # 3. current supply of executors for each node's job; 4. number of remaining tasks in each node.
            # 5. approximate remaining work in each node.
            'dag_batch': Graph(node_space=Box(0, np.inf, (NUM_NODE_FEATURES,)), 
                            edge_space=Discrete(1)),

            # length: num active jobs
            # `ptr[job_idx]` returns the index of the first stage associated 
            # with that job. E.g., the range of stage indices for a job is 
            # given by `ptr[job_idx], ..., (ptr[job_idx+1]-1)`
            # NOTE: upper bound of this space is dynamic, equal to 
            # the number of active stages. Initialized to 1.
            'dag_ptr': Sequence(Discrete(1)),

            # integer that represents how many executors need to be scheduled
            'num_committable_execs': Discrete(num_executors+1),

            # index of job who is releasing executors, if any.
            # set to `self.num_total_jobs` if source is common pool.
            'source_job_idx': Discrete(job_arrival_cap+1),

            # length: num active jobs
            # count of executors associated with each active job,
            # including moving executors and commitments from other jobs
            'exec_supplies': Sequence(Discrete(2*num_executors))
        })



    @property
    def all_jobs_complete(self):
        return self.num_completed_jobs == len(self.jobs.keys())


    @property
    def num_completed_jobs(self):
        return len(self.completed_job_ids)


    @property
    def num_active_jobs(self):
        return len(self.active_job_ids)


    @property
    def info(self):
        return {
            'wall_time': self.wall_time
        }
    

    @property
    def avg_job_duration(self):
        return np.mean(self.job_duration_buff) * 1e-3


    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # simulation wall time in ms
        self.wall_time = 0

        self.datagen.reset(self.np_random)
        self.task_duration_gen.reset(self.np_random)

        # priority queue of scheduling events, indexed by wall time
        try:
            time_limit = options['time_limit']
        except:
            assert self.job_arrival_cap is not None, \
                'must either have a limit on job arrivals or time.'
            time_limit = np.inf
        self.timeline = self.datagen.new_timeline(time_limit)

        # timeline is initially filled with all the job arrival events, so 
        # extract all the job objects from there
        self.jobs = {
            i: event.data['job'] 
            for i, event in enumerate(self.timeline.events())}

        self.executors = [Executor(i) for i in range(self.num_executors)]
        self.exec_tracker.reset()

        # a fast way of obtaining the edge links for an observation is to 
        # start out with all of them in a big array, and then to induce a 
        # subgraph based on the current set of active nodes
        self._reset_edge_links()
        self.num_total_stages = self.all_job_ptr[-1]

        # must be ordered
        self.active_job_ids = []

        self.completed_job_ids = set()

        # maintains the stages that have already been selected during the 
        # current scheduling round so that they don't get selected again 
        # until the next round
        self.selected_stages = set()

        # (index of an active stage) -> (stage object)
        # used to trace an action to its corresponding stage
        self.stage_selection_map = {}

        self._load_initial_jobs()

        return self._observe(), self.info



    def step(self, action):
        self._take_action(action)

        if self.exec_tracker.num_committable_execs() > 0 \
            and len(self.schedulable_stages) > 0:
            # there are still scheduling decisions to be made, so consult 
            # the agent again
            return self._observe(), 0, False, False, self.info
            
        # commitment round has completed, now schedule the free executors
        self._commit_remaining_executors()
        self._fulfill_commitments_from_source()
        self.exec_tracker.clear_executor_source()
        self.selected_stages.clear()

        # save old state attributes for computing reward
        wall_time_step = self.wall_time
        active_job_ids_step = self.active_job_ids.copy()

        # step through timeline until next scheduling event
        self._resume_simulation()

        job_time = self._get_jobtime(wall_time_step, active_job_ids_step)
        reward = -job_time
        terminated = self.all_jobs_complete

        if not terminated:
            assert self.exec_tracker.num_committable_execs() > 0 \
                and len(self.schedulable_stages) > 0

        if self.render_mode == 'human':
            self._render_frame()

        # if the episode isn't done, then start a new scheduling 
        # round at the current executor source
        return self._observe(), reward, terminated, False, self.info



    def close(self):
        if self.renderer:
            self.renderer.close()




    ## internal methods

    def _reset_edge_links(self):
        edge_links = []
        job_ptr = [0]
        for job in self.jobs.values():
            base_stage_idx = job_ptr[-1]
            edges = np.vstack(job.dag.edges)
            edge_links += [base_stage_idx + edges]
            job_ptr += [base_stage_idx + job.num_stages]
        self.all_edge_links = np.vstack(edge_links)
        self.all_job_ptr = np.array(job_ptr)



    def _load_initial_jobs(self):
        while not self.timeline.empty:
            wall_time, event = self.timeline.peek()
            if wall_time > 0:
                break
            self.timeline.pop()
            try:
                self._handle_job_arrival(event.data['job'])
            except:
                raise Exception('initial timeline must only contain jobs')

        self.schedulable_stages = self._find_schedulable_stages()



    def _take_action(self, action):
        assert self.action_space.contains(action), 'invalid action'

        if action['stage_idx'] == -1:
            # no stage has been selected
            self._commit_remaining_executors()
            return

        stage = self.stage_selection_map[action['stage_idx']]
        assert stage in self.schedulable_stages, \
            'the selected stage is not currently schedulable'

        num_executors = action['num_exec']
        assert num_executors >= 1, 'too few executors selected'
        assert num_executors <= self.exec_tracker.num_committable_execs(), \
            'too many executors selected'

        # agent may have requested more executors than are actually needed 
        # or available
        num_executors = self._adjust_num_executors(num_executors, stage)
        self.exec_tracker.add_commitment(num_executors, stage.pool_key)

        # mark stage as selected so that it doesn't get selected again during 
        # this scheduling round
        self.selected_stages.add(stage)

        # find remaining schedulable stages
        job_ids = [stage.job_id for stage in self.schedulable_stages]
        i = bisect_left(job_ids, stage.job_id)
        hi = min(len(job_ids), i + len(self.jobs[stage.job_id].active_stages))
        j = bisect_right(job_ids, stage.job_id, lo=i, hi=hi)
        self.schedulable_stages = \
            self.schedulable_stages[:i] + \
            self._find_schedulable_stages([stage.job_id]) + \
            self.schedulable_stages[j:]



    def _resume_simulation(self):
        '''resumes the simulation until either there are new scheduling 
        decisions to be made, or it's done.
        '''
        assert not self.timeline.empty
        schedulable_stages = []

        while not self.timeline.empty:
            self.wall_time, event = self.timeline.pop()
            self.handle_event[event.type](**event.data)

            if self.exec_tracker.num_committable_execs() == 0:
                continue

            schedulable_stages = self._find_schedulable_stages()
            if len(schedulable_stages) > 0:
                # there are schedulable stages and committable executors,
                # so we need to enter the scheduling loop
                break

            self._move_idle_executors()
            self.exec_tracker.clear_executor_source()

        self.schedulable_stages = schedulable_stages



    def _observe(self):
        self.stage_selection_map.clear()
        
        nodes = []
        dag_ptr = [0]
        active_stage_mask = np.zeros(self.num_total_stages, dtype=bool)
        exec_supplies = []
        source_job_idx = len(self.active_job_ids)

        for i, stage in enumerate(iter(self.schedulable_stages)):
            self.stage_selection_map[i] = stage
            stage.is_schedulable = True

        for i, job_id in enumerate(self.active_job_ids):
            job = self.jobs[job_id]

            if job_id == self.exec_tracker.source_job_id():
                source_job_idx = i

            exec_supplies += [self.exec_tracker.exec_supply(job_id)]

            for stage in job.active_stages:
                nodes += [
                    (
                        stage.num_remaining_tasks, 
                        stage.most_recent_duration, 
                        stage.is_schedulable
                    )
                ]
                stage.is_schedulable = False

                active_stage_mask[self.all_job_ptr[job_id] + stage.id_] = 1

            dag_ptr += [len(nodes)]

        try:
            nodes = np.vstack(nodes).astype(np.float32)
        except:
            # there are no active stages
            nodes = np.zeros((0, NUM_NODE_FEATURES), dtype=np.float32)

        edge_links = graph_utils.subgraph(self.all_edge_links, 
                                          active_stage_mask)

        # not using edge data, so this array is always zeros
        edges = np.zeros(len(edge_links), dtype=int)

        num_committable_execs = self.exec_tracker.num_committable_execs()

        obs = {
            'dag_batch': GraphInstance(nodes, edges, edge_links),
            'dag_ptr': dag_ptr,
            'num_committable_execs': num_committable_execs,
            'source_job_idx': source_job_idx,
            'exec_supplies': exec_supplies
        }

        # update stage action space to reflect the current number of active 
        # stages
        self.observation_space['dag_ptr'].feature_space.n = len(nodes) + 1
        self.action_space['stage_idx'].n = len(nodes) + 1

        return obs
    


    def _render_frame(self):
        executor_histories = (executor.history for executor in self.executors)
        job_completion_times = (
            self.jobs[job_id].t_completed 
            for job_id in iter(self.completed_job_ids)
        )
        average_job_duration = int(metrics.avg_job_duration(self) * 1e-3)

        self.renderer.render_frame(
            executor_histories, 
            job_completion_times,
            self.wall_time,
            average_job_duration,
            self.num_active_jobs,
            self.num_completed_jobs
        )



    ## event handlers

    def _handle_job_arrival(self, job):
        self.active_job_ids += [job.id_]
        self.exec_tracker.add_job(job.id_)
        [self.exec_tracker.add_stage(*stage.pool_key) for stage in job.stages]

        if self.exec_tracker.common_pool_has_executors():
            # if there are any executors that don't belong to any job, then 
            # the agent might want to schedule them to this job, so start a 
            # new round at the common pool
            self.exec_tracker.update_executor_source(COMMON_POOL_KEY)



    def _handle_executor_arrival(self, executor, stage):
        '''performs some bookkeeping when a executor arrives'''
        job = self.jobs[stage.job_id]

        job.add_local_executor(executor)
        executor.add_history(self.wall_time, job.id_)

        self.exec_tracker.count_executor_arrival(stage.pool_key)
        self.exec_tracker.move_executor_to_pool(executor.id_, job.pool_key)

        self._move_executor_to_stage(executor, stage)



    def _handle_task_completion(self, stage, task):
        '''performs some bookkeeping when a task completes'''
        job = self.jobs[stage.job_id]

        executor = self.executors[task.executor_id]
        
        assert not stage.completed
        stage.add_task_completion()
        task.t_completed = self.wall_time
        executor.is_executing = False
        
        if stage.num_remaining_tasks > 0:
            # reassign the executor to keep working on this stage if there is 
            # more work to do
            self._execute_next_task(executor, stage)
            return

        did_job_frontier_change = False

        if stage.completed:
            did_job_frontier_change = self._process_stage_completion(stage)

        if job.completed:
            self._process_job_completion(job)

        # executor may have somewhere to be moved
        had_commitment = \
            self._handle_released_executor(
                executor, 
                stage, 
                did_job_frontier_change
            )

        # executor source may need to be updated
        self._update_executor_source(stage, 
                                     had_commitment, 
                                     did_job_frontier_change)




    ## Other helper functions

    def _commit_remaining_executors(self):
        '''There may be executors at the current source pool that weren't 
        committed anywhere, e.g. because there were no more stages to 
        schedule, or because the agent chose not to schedule all of them.
        
        This function explicitly commits those remaining executors to the 
        common pool. When those executors get released, they either move to 
        the job pool or the common pool, depending on whether the job is 
        saturated at that time. 
        
        It is important to do this, or else the agent could go in a lostage, 
        under-committing executors from the same source pool.
        '''
        num_uncommitted_executors = self.exec_tracker.num_committable_execs()

        if num_uncommitted_executors > 0:
            self.exec_tracker.add_commitment(num_uncommitted_executors, 
                                             COMMON_POOL_KEY)



    def _find_schedulable_stages(self, job_ids=None, source_job_id=None):
        '''An stage is schedulable if it is ready (see `_is_stage_ready()`), 
        it hasn't been selected in the current scheduling round, and its job
        is not saturated with executors (i.e. can accept more executors).
        
        returns a union of schedulable stages over all the jobs specified 
        in `job_ids`. If no job ids are provided, then all active jobs are 
        searched.
        '''
        if job_ids is None:
            job_ids = list(self.active_job_ids)

        if source_job_id is None:
            source_job_id = self.exec_tracker.source_job_id()

        # filter out saturated jobs. The source job is never considered 
        # saturated, because it is not gaining any new executors during 
        # scheduling
        job_ids = [
            job_id \
            for job_id in job_ids
            if job_id == source_job_id or \
               self.exec_tracker.exec_supply(job_id) \
                < self.num_executors
        ]

        schedulable_stages = [
            stage \
            for job_id in iter(job_ids) \
            for stage in iter(self.jobs[job_id].active_stages)
            if stage not in self.selected_stages \
                and self._is_stage_ready(stage)
        ]

        return schedulable_stages



    def _is_stage_ready(self, stage):
        '''a stage is ready if 
        - it is unsaturated, and
        - all of its parent stages are saturated
        '''
        if self._is_stage_saturated(stage):
            return False

        job = self.jobs[stage.job_id]
        for parent_stage in job.parent_stages(stage):
            if not self._is_stage_saturated(parent_stage):
                return False

        return True



    def _adjust_num_executors(self, num_executors, stage):
        '''truncates the numer of executor assigned to `stage` to the stage's
        demand, if it's larger
        '''
        executor_demand = self._get_executor_demand(stage)
        num_executors_adjusted = min(num_executors, executor_demand)
        assert num_executors_adjusted > 0
        return num_executors_adjusted



    def _get_executor_demand(self, stage):
        '''a stage's executor demand is the number of executors that it can 
        accept in addition to the executors currently working on, committed to,
        and moving to the stage. Note: demand can be negative if more 
        resources were assigned to the stage than needed.
        '''
        num_executors_moving = \
            self.exec_tracker.num_executors_moving_to_stage(stage.pool_key)
        num_commitments = \
            self.exec_tracker.num_commitments_to_stage(stage.pool_key)

        demand = stage.num_remaining_tasks \
            - (num_executors_moving + num_commitments)
        return demand



    def _is_stage_saturated(self, stage):
        '''a stage is saturated if it doesn't need any more executors.'''
        return self._get_executor_demand(stage) <= 0
            


    def _execute_next_task(self, executor, stage):
        '''starts work on another one of `stage`'s tasks, assuming there are 
        still tasks remaining and the executor is local to the stage
        '''
        assert stage.num_remaining_tasks > 0
        assert executor.is_at_job(stage.job_id)
        assert not executor.is_executing

        job = self.jobs[stage.job_id]

        task = stage.start_on_next_task()
        if stage.num_remaining_tasks == 0:
            # stage just became saturated
            job.saturated_stage_count += 1

        num_local_executors = len(self.jobs[stage.job_id].local_executors)
        task_duration = \
            self.task_duration_gen.sample(
                task, 
                executor, 
                num_local_executors, 
                stage.task_duration_data
            )

        executor.task = task
        executor.is_executing = True
        task.executor_id = executor.id_
        task.t_accepted = self.wall_time
        stage.most_recent_duration = task_duration

        self.timeline.push(
            self.wall_time + task_duration, 
            TimelineEvent(
                type = TimelineEvent.Type.TASK_COMPLETION, 
                data = {'stage': stage, 'task': task}
            )
        )



    def _send_executor(self, executor, stage):
        '''sends a `executor` to `stage`, assuming that the executor is 
        currently at a different job
        '''
        assert stage is not None
        assert not executor.is_executing
        assert not executor.is_at_job(stage.job_id)

        self.exec_tracker.move_executor_to_pool(executor.id_, 
                                                stage.pool_key, 
                                                send=True)

        if executor.job_id is not None:
            old_job = self.jobs[executor.job_id]
            old_job.remove_local_executor(executor)

        self.timeline.push(
            self.wall_time + self.moving_delay, 
            TimelineEvent(
                type = TimelineEvent.Type.EXECUTOR_ARRIVAL, 
                data = {'executor': executor, 'stage': stage}
            )
        )
        


    def _handle_released_executor(
        self, 
        executor, 
        stage, 
        did_job_frontier_change
    ):
        '''called upon a task completion. if the executor has been commited to 
        a next stage, then try assigning it there. Otherwise, if `stage` became
        saturated and unlocked new stages within its job dag, then move the 
        executor to the job's executor pool so that it can be assigned to the 
        new stages
        '''
        commitment_pool_key = self.exec_tracker.peek_commitment(stage.pool_key)

        if commitment_pool_key is not None:
            self._fulfill_commitment(executor.id_, commitment_pool_key)
            return True
        
        # executor has nowhere to go, so make it idle
        executor.task = None
    
        if did_job_frontier_change:
            self._move_idle_executors(stage.pool_key, [executor.id_])

        return False

        


    def _update_executor_source(
        self, 
        stage, 
        had_commitment, 
        did_job_frontier_change
    ):
        '''called upon a task completion. If any new stages were unlocked 
        within this job upon the task completion, then start a new commitment 
        round at this job's pool so that free executors can be assigned to the 
        new stages. Otherwise, if the executor has nowhere to go, then start a 
        new commitment round at this stage's pool to give it somewhere to go.
        '''
        if did_job_frontier_change:
            self.exec_tracker.update_executor_source(stage.job_pool_key)
        elif not had_commitment:
            self.exec_tracker.update_executor_source(stage.pool_key)



    def _process_stage_completion(self, stage):
        '''performs some bookkeeping when a stage completes'''
        job = self.jobs[stage.job_id]
        frontier_changed = job.add_stage_completion(stage)
        return frontier_changed
        

    
    def _process_job_completion(self, job):
        '''performs some bookkeeping when a job completes'''
        assert job.id_ in self.jobs

        # if there are any executors still local to this job, then remove them
        if len(self.exec_tracker._pools[job.pool_key]) > 0:
            self._move_idle_executors(job.pool_key)

        assert len(self.exec_tracker._pools[job.pool_key]) == 0
        
        self.active_job_ids.remove(job.id_)
        self.completed_job_ids.add(job.id_)
        job.t_completed = self.wall_time
        self.job_duration_buff.append(job.t_completed - job.t_arrival)



    def _fulfill_commitment(self, executor_id, dst_pool_key):
        src_pool_key = self.exec_tracker.remove_commitment(executor_id, 
                                                           dst_pool_key)

        if dst_pool_key == COMMON_POOL_KEY:
            # this executor is free and isn't commited to any actual stage
            self._move_idle_executors(src_pool_key, [executor_id])
            return

        job_id, stage_id = dst_pool_key
        stage = self.jobs[job_id].stages[stage_id]
        executor = self.executors[executor_id]

        self._move_executor_to_stage(executor, stage)



    def _get_idle_source_executors(self, pool_key=None):
        if not pool_key:
            executor_ids = self.exec_tracker.get_source_executors()
        else:
            executor_ids = self.exec_tracker._pools[pool_key]

        free_executor_ids = set((
            executor_id
            for executor_id in iter(executor_ids)
            if not self.executors[executor_id].is_executing
        ))

        return free_executor_ids
        


    def _fulfill_commitments_from_source(self):
        # only consider the idle executors
        idle_executor_ids = self._get_idle_source_executors()
        commitments = self.exec_tracker.get_source_commitments()

        for dst_pool_key, num_executors in commitments.items():
            assert num_executors > 0
            while num_executors > 0 and len(idle_executor_ids) > 0:
                executor_id = idle_executor_ids.pop()
                self._fulfill_commitment(executor_id, dst_pool_key)
                num_executors -= 1

        assert len(idle_executor_ids) == 0



    def _move_idle_executors(self, src_pool_key=None, executor_ids=None):
        '''When an executor becomes idle, it may need to be moved somewhere. 
        If it's idle at a stage, it might need to be moved to the job pool. 
        If it's idle at a job, it might need to be moved to the common pool.
        '''
        if src_pool_key is None:
            src_pool_key = self.exec_tracker.get_source()
        assert src_pool_key is not None

        if src_pool_key == COMMON_POOL_KEY:
            return # no-op

        if executor_ids is None:
            executor_ids = list(self._get_idle_source_executors(src_pool_key))
        assert len(executor_ids) > 0

        job_id, stage_id = src_pool_key
        is_job_saturated = self.jobs[job_id].saturated
        if stage_id is None and not is_job_saturated:
            # source is an unsaturated job's pool
            return # no-op

        # if the source is a saturated job's pool, then move it to the common 
        # pool. If it's a stage's pool, then move it to the job's pool.
        dst_pool_key = COMMON_POOL_KEY if is_job_saturated else (job_id, None)

        for executor_id in executor_ids:
            self.exec_tracker.move_executor_to_pool(executor_id, dst_pool_key)
            if dst_pool_key == COMMON_POOL_KEY:
                executor = self.executors[executor_id]
                job = self.jobs[job_id]
                job.remove_local_executor(executor)
                executor.add_history(self.wall_time, -1)



    def _try_backup_schedule(self, executor):
        '''If a executor arrives to a stage that no longer needs any executors, 
        then greedily try to find a backup stage.
        '''
        backup_stage = self._find_backup_stage(executor)
        if backup_stage:
            # found a backup
            self._move_executor_to_stage(executor, backup_stage)
            return

        # no backup stage found, so move executor to job or common pool 
        # depending on whether or not the executor's job is saturated
        exec_location = self.exec_tracker._executor_locations[executor.id_]
        self._move_idle_executors(exec_location, [executor.id_])



    def _move_executor_to_stage(self, executor, stage):
        if stage.num_remaining_tasks == 0:
            # stage is saturated, so this executor is not needed there anymore
            self._try_backup_schedule(executor)
            return

        if not executor.is_at_job(stage.job_id):
            self._send_executor(executor, stage)
            return

        job = self.jobs[stage.job_id]
        if stage not in job.frontier_stages:
            # stage is not ready yet; make executor idle and move it to the
            # job pool
            executor.task = None
            self.exec_tracker.move_executor_to_pool(executor.id_, 
                                                    stage.job_pool_key)
            return

        # stage's dependencies are satisfied, so start working on it.
        self.exec_tracker.move_executor_to_pool(executor.id_, stage.pool_key)
        self._execute_next_task(executor, stage)



    def _find_backup_stage(self, executor):
        # first, try searching within the same job
        local_stages = \
            self._find_schedulable_stages(
                job_ids=[executor.job_id], 
                source_job_id=executor.job_id
            )
        
        if len(local_stages) > 0:
            return local_stages.pop()

        # now, try searching all other jobs
        other_job_ids = [
            job_id 
            for job_id in iter(self.active_job_ids)
            if not executor.is_at_job(job_id)
        ]
        
        other_stages = \
            self._find_schedulable_stages(
                job_ids=other_job_ids,
                source_job_id=executor.job_id
            )

        if len(other_stages) > 0:
            return other_stages.pop()

        # out of luck
        return None



    def _get_jobtime(self, wall_time_step, active_job_ids_step):
        duration = self.wall_time - wall_time_step
        if duration == 0:
            return 0

        # include jobs that completed and arrived during the most recent 
        # simulation run
        all_job_ids = set(active_job_ids_step) | set(self.active_job_ids)

        job_time = 0
        for job_id in iter(all_job_ids):
            job = self.jobs[job_id]
            start = max(job.t_arrival, wall_time_step)
            end = min(job.t_completed, self.wall_time)

            if self.beta == 0:
                job_time += end - start
            else:
                # continuously discounted job-time
                job_time += \
                    np.exp(-self.beta * 1e-3 * (start - wall_time_step)) - \
                    np.exp(-self.beta * 1e-3 * (end - wall_time_step))

        if self.beta > 0:
            job_time /= self.beta

        return job_time