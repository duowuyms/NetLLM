import numpy as np
import networkx as nx

from typing import List
from .stage import Stage


class Job:
    '''An object representing a job in the system, containing a set of stages with dependencies stored in a dag.'''

    def __init__(
        self, 
        id_: int, 
        stages: List[Stage], 
        dag: nx.DiGraph, 
        t_arrival: float,
        query_size: int,
        query_num: int
    ):
        # unique identifier of this job
        self.id_ = id_

        # list of objects of all the stages
        # that belong to this job
        # TODO: use ordered set
        self.stages = stages

        # all incomplete stages
        # TODO: use ordered set
        self.active_stages = stages.copy()
        
        # incomplete stages whose parents have completed
        self.frontier_stages = set()

        # networkx dag storing the stage dependencies
        self.dag = dag

        # time that this job arrived into the system
        self.t_arrival = t_arrival

        self.query_size = query_size

        self.query_num = query_num

        # time that this job completed, i.e. when the last
        # stage completed
        self.t_completed = np.inf

        # set of executors that are local to this job
        self.local_executors = set()

        # count of stages who have no remaining tasks
        self.saturated_stage_count = 0

        self.init_frontier()


    def __str__(self):
         return f'TPCH_{self.query_num}_{self.query_size}'



    @property
    def pool_key(self):
        return (self.id_, None)



    @property
    def completed(self):
        return self.num_active_stages == 0



    @property
    def saturated(self):
        return self.saturated_stage_count == len(self.stages)



    @property
    def num_stages(self):
        return len(self.stages)



    @property
    def num_active_stages(self):
        return len(self.active_stages)



    def add_stage_completion(self, stage):
        '''increments the count of completed stages'''
        self.active_stages.remove(stage)

        self.frontier_stages.remove(stage)

        new_stages = self.find_new_frontier_stages(stage)
        self.frontier_stages |= new_stages

        return len(new_stages) > 0
            


    def init_frontier(self):
        '''returns a set containing all the stages which are
        source nodes in the dag, i.e. which have no dependencies
        '''
        assert len(self.frontier_stages) == 0
        self.frontier_stages |= self.source_stages()



    def source_stages(self):
        return set(
            self.stages[node]
            for node, in_deg in self.dag.in_degree()
            if in_deg == 0
        )



    def children_stages(self, stage):
        return (self.stages[stage_id] for stage_id in self.dag.successors(stage.id_))



    def parent_stages(self, stage):
        return (self.stages[stage_id] for stage_id in self.dag.predecessors(stage.id_))



    def find_new_frontier_stages(self, stage):
        '''if ` stage` is completed, returns all of its successors whose other dependencies are also 
        completed, if any exist.
        '''
        if not stage.completed:
            return set()

        new_stages = set()
        # search through stage's children
        for suc_stage_id in self.dag.successors(stage.id_):
            # if all dependencies are satisfied, then add this child to the frontier
            new_stage = self.stages[suc_stage_id]
            if not new_stage.completed and self.check_dependencies(suc_stage_id):
                new_stages.add(new_stage)
        
        return new_stages



    def check_dependencies(self, stage_id):
        '''searches to see if all the dependencies of stage with id `stage_id` are satisfied.'''
        for dep_id in self.dag.predecessors(stage_id):
            if not self.stages[dep_id].completed:
                return False

        return True



    def add_local_executor(self, executor):
        assert executor.task is None
        self.local_executors.add(executor.id_)
        executor.job_id = self.id_



    def remove_local_executor(self, executor):
        self.local_executors.remove(executor.id_)
        executor.job_id = None
        executor.task = None