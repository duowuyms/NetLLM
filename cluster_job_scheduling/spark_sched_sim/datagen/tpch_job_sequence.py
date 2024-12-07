
import os
import os.path as osp
import pathlib
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
import networkx as nx

from .base_job_sequence import BaseJobSequenceGen
from ..components import Job, Stage



TPCH_URL = 'https://bit.ly/3F1Go8t'
QUERY_SIZES = ['2g','5g','10g','20g','50g','80g','100g']
NUM_QUERIES = 22


class TPCHJobSequenceGen(BaseJobSequenceGen):
    def __init__(self, job_arrival_rate, job_arrival_cap):
        super().__init__(job_arrival_rate, job_arrival_cap)
        self.cwd = os.getcwd()
        if 'cluster_job_scheduling' not in self.cwd:
            self.cwd = osp.join(self.cwd, 'cluster_job_scheduling')
        if not osp.isdir(osp.join(self.cwd, 'data/tpch')):
            self.download()
            

    def download(self):
        print('Downloading the TPC-H dataset...', flush=True)
        # pathlib.Path('data/tpch').mkdir(parents=True, exist_ok=True) 
        pathlib.Path(osp.join(self.cwd, 'data')).mkdir(parents=True, exist_ok=True) 
        with urlopen(TPCH_URL) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('data')
        print('Done.', flush=True)


    def generate_job(self, job_id, t_arrival):
        query_num = 1 + self.np_random.integers(NUM_QUERIES)
        query_size = self.np_random.choice(QUERY_SIZES)
        query_path = osp.join(osp.join(self.cwd, 'data/tpch'), str(query_size))
        
        adj_matrix = np.load(
            osp.join(query_path, f'adj_mat_{query_num}.npy'), 
            allow_pickle=True)
        
        task_durations = np.load(
            osp.join(query_path, f'task_duration_{query_num}.npy'), 
            allow_pickle=True).item()
        
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        assert adj_matrix.shape[0] == len(task_durations)

        num_stages = adj_matrix.shape[0]
        stages = []
        for stage_id in range(num_stages):
            task_duration_data = task_durations[stage_id]
            e = next(iter(task_duration_data['first_wave']))

            num_tasks = len(task_duration_data['first_wave'][e]) + \
                        len(task_duration_data['rest_wave'][e])

            # remove fresh duration from first wave duration
            # drag nearest neighbor first wave duration to empty spots
            self._pre_process_task_duration(task_duration_data)

            # generate a node
            stages += [Stage(
                stage_id, 
                job_id, 
                num_tasks, 
                task_duration_data
            )]

        # generate DAG
        dag = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        for _,_,d in dag.edges(data=True):
            d.clear()
        
        return Job(job_id, stages, dag, t_arrival, query_size, query_num)



    def _pre_process_task_duration(self, task_duration):
        # remove fresh durations from first wave
        clean_first_wave = {}
        for e in task_duration['first_wave']:
            clean_first_wave[e] = []
            fresh_durations = MultiSet()
            # O(1) access
            for d in task_duration['fresh_durations'][e]:
                fresh_durations.add(d)
            for d in task_duration['first_wave'][e]:
                if d not in fresh_durations:
                    clean_first_wave[e].append(d)
                else:
                    # prevent duplicated fresh duration blocking first wave
                    fresh_durations.remove(d)

        # fill in nearest neighour first wave
        last_first_wave = []
        for e in sorted(clean_first_wave.keys()):
            if len(clean_first_wave[e]) == 0:
                clean_first_wave[e] = last_first_wave
            last_first_wave = clean_first_wave[e]

        # swap the first wave with fresh durations removed
        task_duration['first_wave'] = clean_first_wave




class MultiSet(object):
    """
    allow duplication in set
    """
    def __init__(self):
        self.set = {}

    def __contains__(self, item):
        return item in self.set

    def add(self, item):
        if item in self.set:
            self.set[item] += 1
        else:
            self.set[item] = 1

    def clear(self):
        self.set.clear()

    def remove(self, item):
        self.set[item] -= 1
        if self.set[item] == 0:
            del self.set[item]