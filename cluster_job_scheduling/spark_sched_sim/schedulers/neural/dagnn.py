import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as gnn

from .neural import *
from spark_sched_sim.wrappers import DAGNNObsWrapper


class DAGNNScheduler(NeuralScheduler):
    '''Scheduler that uses DAGNN architecture. Asyncronous message passing
    with attention and GRU cells
    Paper: https://arxiv.org/abs/2101.07965
    '''
    
    def __init__(
        self,
        num_executors,
        embed_dim,
        num_encoder_layers,
        policy_mlp_kwargs,
        state_dict_path=None,
        opt_cls=None,
        opt_kwargs=None,
        max_grad_norm=None,
        num_node_features=5,
        num_dag_features=3,
    ):
        name = 'DAGNN'
        if state_dict_path:
            name += f':{state_dict_path}'

        actor = ActorNetwork(
            num_executors, num_node_features, num_dag_features, 
            embed_dim, num_encoder_layers, policy_mlp_kwargs)
        
        obs_wrapper_cls = DAGNNObsWrapper

        super().__init__(
            name,
            actor,
            obs_wrapper_cls,
            num_executors,
            state_dict_path,
            opt_cls,
            opt_kwargs,
            max_grad_norm
        )

    

class ActorNetwork(nn.Module):
    def __init__(
        self, 
        num_executors, 
        num_node_features, 
        num_dag_features, 
        embed_dim, 
        num_encoder_layers,
        policy_mlp_kwargs
    ):
        super().__init__()
        self.encoder = EncoderNetwork(
            num_node_features, embed_dim, num_encoder_layers)

        emb_dims = {
            'node': embed_dim * num_encoder_layers,
            'dag': embed_dim,
            'glob': embed_dim
        }

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, policy_mlp_kwargs)

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, policy_mlp_kwargs)



class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, embed_dim, num_layers):
        super().__init__()
        self.node_encoder = NodeEncoder(num_node_features, embed_dim, num_layers)
        self.dag_encoder = DagEncoder(embed_dim * num_layers, embed_dim)
        self.global_encoder = GlobalEncoder(embed_dim)

    def forward(self, dag_batch):
        '''
            Returns:
                a dict of representations at three different levels:
                node, dag, and global.
        '''
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        try:
            # batch of obsns
            obs_ptr = dag_batch['obs_ptr']
            h_glob = self.global_encoder(h_dag, obs_ptr)
        except:
            # single obs
            h_glob = self.global_encoder(h_dag)

        h_dict = {
            'node': h_node, 
            'dag': h_dag,   
            'glob': h_glob  
        }

        return h_dict
    


class AttnConv(gnn.MessagePassing):
    def __init__(self, emb_dim, reverse_flow):
       flow = 'target_to_source' if reverse_flow else 'source_to_target'
       super().__init__(flow=flow)
       self.attn_layer = nn.Sequential(
           nn.Linear(2 * emb_dim, 1), 
           nn.LeakyReLU(negative_slope=.2, inplace=True))

    def forward(self, hs, edge_index):
       h, h_prev = hs
       return self.propagate(edge_index, h=h, h_prev=h_prev)

    def message(self, h_j, h_prev_i, index, size_i):
       q, k, v = h_prev_i, h_j, h_j
       alpha_j = self.attn_layer(torch.cat([q, k], -1))
       alpha_j = pyg_utils.softmax(alpha_j, index=index, num_nodes=size_i)
       return alpha_j * v



class NodeEncoder(nn.Module):
    def __init__(self, num_node_features, embed_dim, num_layers, reverse_flow=True):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.lin_init = nn.Linear(num_node_features, embed_dim)

        self.attn_convs = nn.ModuleList(
            [AttnConv(embed_dim, reverse_flow) for _ in range(num_layers)])
        
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(embed_dim, embed_dim) for _ in range(num_layers)])


    def forward(self, dag_batch):
        edge_masks = dag_batch['edge_masks']

        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)

        # stores node representations from the previous layer
        h_prev = self.lin_init(dag_batch.x)
        
        # stores node representations from each layer
        h_list = []

        num_nodes = h_prev.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes)

        for attn_conv, gru_cell in zip(self.attn_convs, self.gru_cells):
            h = torch.zeros_like(h_prev)

            # first process the source nodes which have no incoming messages
            h[src_node_mask] = gru_cell(h_prev[src_node_mask])

            edge_masks_it = iter(reversed(edge_masks)) \
                if self.reverse_flow else iter(edge_masks)

            # pass messages one level of the dag (batch) at a time
            for edge_mask in edge_masks_it:
                edge_index_masked = dag_batch.edge_index[:, edge_mask]
                dst_mask = pyg_utils.index_to_mask(
                    edge_index_masked[self.i], num_nodes)

                msg = attn_conv((h.clone(), h_prev), edge_index_masked)
                h[dst_mask] = gru_cell(h_prev[dst_mask], msg[dst_mask])

            h_list += [h.clone()]
            h_prev = h

        # concatenate the representations across layers
        h = torch.cat(h_list, 1)

        return h
    
    
    def _forward_no_mp(self, x):
        '''forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        '''
        h_prev = self.lin_init(x)

        h_list = []
        for gru_cell in self.gru_cells:
            h = gru_cell(h_prev)
            h_list += [h.clone()]
            h_prev = h

        return torch.cat(h_list, 1)



class DagEncoder(nn.Module):
    def __init__(self, dim_node_emb, embed_dim):
        super().__init__()
        self.lin = nn.Linear(dim_node_emb, embed_dim)

    def forward(self, h_node, dag_batch):
        '''max pool over terminal node representations for each dag'''
        terminal_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[1], dag_batch.x.shape[0])
        
        # readout using only the terminal nodes
        h_dag = gnn.global_max_pool(
            h_node[terminal_node_mask], 
            dag_batch.batch[terminal_node_mask], 
            size=dag_batch.num_graphs)

        return self.lin(h_dag)
    


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, h_dag, obs_ptr=None):
        '''max pool over dag representations for each obs'''
        if obs_ptr is not None:
            # batch of obsns
            h_glob = segment_csr(h_dag, obs_ptr, reduce='max')
        else:
            # single obs
            h_glob = h_dag.max(0)[0].unsqueeze(0)

        return self.lin(h_glob)
        