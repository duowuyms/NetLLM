"""
Customized state encoder based on Decima's encoder.
"""
import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse

import spark_sched_sim.graph_utils as graph_utils


class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_dim, embed_dim):
        super().__init__()

        self.node_encoder = NodeEncoder(num_node_features, hidden_dim, embed_dim)
        
        self.dag_encoder = DagEncoder(num_node_features, hidden_dim, embed_dim)
        
        self.global_encoder = GlobalEncoder(hidden_dim, embed_dim)


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



class NodeEncoder(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        hidden_dim,
        embed_dim, 
        reverse_flow=True
    ):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.mlp_msg = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )


    def forward(self, dag_batch):
        edge_masks = dag_batch['edge_masks']
        
        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)
        
        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes)
        
        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        edge_masks_it = iter(reversed(edge_masks)) \
            if self.reverse_flow else iter(edge_masks)

        # target-to-source message passing, one level of the dags at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = graph_utils.make_adj(edge_index_masked, num_nodes)

            # nodes sending messages
            src_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.j], num_nodes)

            # nodes receiving messages
            dst_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(
                adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])

        return h
    

    def _forward_no_mp(self, x):
        '''forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        '''
        return self.mlp_prep(x)
    


class DagEncoder(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        hidden_dim,
        embed_dim, 
    ):
        super().__init__()
        input_dim = num_node_features + embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, h_node, dag_batch):
        # include original input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.mlp(h_node), dag_batch.ptr)
        return h_dag
    


class GlobalEncoder(nn.Module):
    def __init__(self, hidden_dim, embed_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, h_dag, obs_ptr=None):
        h_dag = self.mlp(h_dag)

        if obs_ptr is not None:
            # batch of observations
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob