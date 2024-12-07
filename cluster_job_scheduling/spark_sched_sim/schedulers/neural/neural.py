import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import *
from torch.optim import *
from torch.distributions.utils import clamp_probs
from torch_scatter import segment_csr
from gymnasium.core import ObsType, ActType
import torch_geometric.utils as pyg_utils
import numpy as np

from ..scheduler import Scheduler
from spark_sched_sim import graph_utils



class NeuralScheduler(Scheduler):
    '''Base class for all neural schedulers'''

    def __init__(
        self,
        name,
        actor,
        obs_wrapper_cls,
        num_executors,
        state_dict_path,
        opt_cls,
        opt_kwargs,
        max_grad_norm,
        device='cpu'
    ):
        super().__init__(name)

        self.actor = actor.to(device)
        self.obs_wrapper_cls = obs_wrapper_cls
        self.num_executors = num_executors
        self.opt_cls = opt_cls
        self.opt_kwargs = opt_kwargs
        self.max_grad_norm = max_grad_norm
        self.device = device
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location=device)
            self.actor.load_state_dict(state_dict)
        self.lr_scheduler = None


    def train(self):
        '''call only on an instance that is about to be trained'''
        assert self.opt_cls, 'optimizer was not specified.'
        self.actor.train()
        glob = globals()
        assert self.opt_cls in glob, \
            f"'{self.opt_cls}' is not a valid optimizer."
        opt_kwargs = self.opt_kwargs or {}
        self.optim = glob[self.opt_cls](
            self.actor.parameters(), **opt_kwargs)
        self.lr_scheduler = lr_scheduler.StepLR(self.optim, 100, .5)


    @torch.no_grad()
    def schedule(self, obs: ObsType) -> ActType:
        dag_batch = graph_utils.obs_to_pyg(obs)
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch['stage_mask']

        dag_batch.to(self.device, non_blocking=True)

        # 1. compute node, dag, and global representations
        h_dict = self.actor.encoder(dag_batch)

        # 2. select a schedulable stage
        stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
        stage_idx, stage_lgprob = self._sample(stage_scores)

        # retrieve index of selected stage's job
        stage_idx_glob = pyg_utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob].item()

        # 3. select the number of executors to add to that stage, conditioned 
        # on that stage's job
        exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_idx)
        num_exec, exec_lgprob = self._sample(exec_scores)

        action = {
            'stage_idx': stage_idx,
            'job_idx': job_idx,
            'num_exec': num_exec
        }

        lgprob = stage_lgprob + exec_lgprob

        return action, lgprob
    

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob


    def evaluate_actions(self, dag_batch, actions):
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        stage_selections, job_indices, exec_selections = \
            [col.clone() for col in actions.T]

        num_stage_acts = dag_batch['num_stage_acts']
        num_exec_acts = dag_batch['num_exec_acts']
        num_nodes_per_obs = dag_batch['num_nodes_per_obs']
        obs_ptr = dag_batch['obs_ptr']
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model with grads enabled
        dag_batch.to(self.device)
        h_dict = self.actor.encoder(dag_batch)
        stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
        exec_scores = self.actor.exec_policy_network(
            dag_batch, h_dict, job_indices)

        stage_lgprobs, stage_entropies = self._evaluate(
            stage_scores.cpu(), num_stage_acts, stage_selections)
        
        exec_lgprobs, exec_entropies = self._evaluate(
            exec_scores.cpu(), num_exec_acts[job_indices], exec_selections)

        # aggregate the evaluations for nodes and dags
        action_lgprobs = stage_lgprobs + exec_lgprobs

        action_entropies = stage_entropies + exec_entropies
        action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return action_lgprobs, action_entropies
    

    @classmethod
    def _evaluate(cls, scores, counts, selections):
        ptr = counts.cumsum(0)
        ptr = torch.cat([torch.tensor([0]), ptr], 0)
        selections += ptr[:-1]
        probs = pyg_utils.softmax(scores, ptr=ptr)
        probs = clamp_probs(probs)
        log_probs = probs.log()
        selection_log_probs = log_probs[selections]
        entropies = -segment_csr(log_probs * probs, ptr)
        return selection_log_probs, entropies


    def update_parameters(self, loss=None):
        if loss:
            # accumulate gradients
            loss.backward()

        if self.max_grad_norm:
            # clip grads
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 
                    self.max_grad_norm,
                    error_if_nonfinite=True
                )
            except:
                print('infinite grad; skipping update.')
                return

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()



def make_mlp(input_dim, hid_dims, output_dim, act_cls, act_kwargs=None):
    if isinstance(act_cls, str):
        glob = globals()
        assert act_cls in glob, f"'{act_cls}' is not a valid activation."
        act_cls = glob[act_cls]

    mlp = nn.Sequential()
    prev_dim = input_dim
    hid_dims = hid_dims + [output_dim]
    for i, dim in enumerate(hid_dims):
        mlp.append(nn.Linear(prev_dim, dim))
        if i == len(hid_dims) - 1:
            break
        act_fn = act_cls(**act_kwargs) if act_kwargs else act_cls()
        mlp.append(act_fn)
        prev_dim = dim
    return mlp



class StagePolicyNetwork(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        emb_dims, 
        mlp_kwargs
    ):
        super().__init__()
        input_dim = num_node_features + \
            emb_dims['node'] + emb_dims['dag'] + emb_dims['glob']

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)


    def forward(self, dag_batch, h_dict):
        # comment of wuduo: the number of activate (schedulable) stages 
        stage_mask = dag_batch['stage_mask']  

        x = dag_batch.x[stage_mask]

        h_node = h_dict['node'][stage_mask]

        batch_masked = dag_batch.batch[stage_mask]
        h_dag_rpt = h_dict['dag'][batch_masked]

        try:
            num_stage_acts = dag_batch['num_stage_acts'] # batch of obsns
        except:
            # comments of wuduo: number of active stages
            num_stage_acts = stage_mask.sum() # single obs 

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0)

        # residual connections to original features
        node_inputs = torch.cat(
            [
                x, 
                h_node, 
                h_dag_rpt, 
                h_glob_rpt
            ], 
            dim=1
        )

        node_scores = self.mlp_score(node_inputs).squeeze(-1)
        return node_scores
    


class ExecPolicyNetwork(nn.Module):
    def __init__(
        self, 
        num_executors, 
        num_dag_features, 
        emb_dims, 
        mlp_kwargs
    ):
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        input_dim = num_dag_features + emb_dims['dag'] + emb_dims['glob'] + 1

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    
    def forward(self, dag_batch, h_dict, job_indices):
        assert isinstance(job_indices, int) or isinstance(job_indices, list) and len(job_indices) == 1
        exec_mask = dag_batch['exec_mask']  # whether each exectutor can be allocated to the job

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, :self.num_dag_features]
        x_dag = x_dag[job_indices]

        h_dag = h_dict['dag'][job_indices]

        exec_mask = exec_mask[job_indices]

        try:
            # batch of obsns
            num_exec_acts = dag_batch['num_exec_acts'][job_indices]
        except:
            # single obs
            num_exec_acts = exec_mask.sum()  # the number of availble executors
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)
            exec_mask = exec_mask.unsqueeze(0)

        exec_actions = self._get_exec_actions(exec_mask)

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)

        x_h_dag_rpt = x_h_dag.repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], 
            dim=0)

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], 
            dim=0)
        
        dag_inputs = torch.cat(
            [
                x_h_dag_rpt,
                h_glob_rpt,
                exec_actions
            ], 
            dim=1
        )

        dag_scores = self.mlp_score(dag_inputs).squeeze(-1)
        return dag_scores
    
    
    def _get_exec_actions(self, exec_mask):
        exec_actions = torch.arange(self.num_executors) / self.num_executors
        exec_actions = exec_actions.to(exec_mask.device)
        exec_actions = exec_actions.repeat(exec_mask.shape[0])
        exec_actions = exec_actions[exec_mask.view(-1)]
        exec_actions = exec_actions.unsqueeze(1)
        return exec_actions