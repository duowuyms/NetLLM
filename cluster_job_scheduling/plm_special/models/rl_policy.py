import enum
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

from collections import deque
    

INF = 1e5


class UseStageHead(enum.Enum):
    BOTH = 1
    HEAD1 = 2  # use stage head 1 (predict next stage like decima)
    HEAD2 = 3  # use stage head 2 (predict next stage in our own way)


class OfflineRLPolicy(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(
            self,
            stage_state_dim,
            exec_state_dim,
            max_stage_num,
            max_exec_num,
            state_encoder,
            plm,
            plm_embed_size,
            max_length=None,
            max_ep_len=4096,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            residual = False, 
            which_layer = -1,  # for early stopping: specify which layer to stop
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.stage_state_dim = stage_state_dim
        self.exec_state_dim = exec_state_dim
        self.max_stage_num = max_stage_num
        self.max_exec_num = max_exec_num
        self.max_length = max_length

        self.state_encoder = state_encoder
        self.plm = plm
        self.plm_embed_size = plm_embed_size

        self.embed_timestep = nn.Embedding(max_ep_len + 1, plm_embed_size).to(device)
        self.embed_return = nn.Linear(1, plm_embed_size).to(device)
        self.embed_stage_state = nn.Linear(stage_state_dim, plm_embed_size).to(device)
        self.embed_exec_state = nn.Linear(exec_state_dim, plm_embed_size).to(device)
        self.embed_action = nn.Linear(2, plm_embed_size).to(device)

        self.embed_ln = nn.LayerNorm(plm_embed_size).to(device)
        
        self.stage_action_head1 = nn.Linear(plm_embed_size, 1).to(device)  # predict the next stage to run in the decima's manner
        self.stage_action_head2 = nn.Linear(plm_embed_size, max_stage_num).to(device)  # predict the next stage to run with direct softmax
        self.exec_action_head = nn.Linear(plm_embed_size, max_exec_num).to(device)  # predict the parallelism limit with direct softmax

        self.device = device
        self.device_out = device_out

        # the following are used for evaluation
        self.states_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.returns_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)
        self.actions_dq = deque([torch.zeros((1, 0, plm_embed_size), device=device)], maxlen=max_length)

        self.residual = residual
        self.which_layer = which_layer
        self.modules_except_plm = nn.ModuleList([  # used to save and load modules except plm
            self.state_encoder, self.embed_timestep, self.embed_return, self.embed_stage_state, self.embed_exec_state,
            self.embed_action, self.embed_ln, self.stage_action_head1, self.stage_action_head2, self.exec_action_head
        ])

    def forward(self, states, actions, returns, timesteps, stage_indices, attention_mask=None, use_head=UseStageHead.HEAD2):
        """
        Forward function, used for training.
        """
        assert actions.shape[0] == 1, 'batch size should be 1, due to the complex structure of DAG information.'

        # Step 1: process actions, returns and timesteps first as they are simple
        actions = actions.to(self.device)  # shape: (1, seq_len, 2)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)

        # 1.1 embed action, return, timestep
        action_embeddings = self.embed_action(actions)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns)  # shape: (1, seq_len, embed_size)
        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)

        # 1.2 time embeddings are treated similar to positional embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Step 2: process states, turn them into embeddings.
        # the challenge here is that the elements in each state is dynamic.
        # this is also the reason why we can only set batch size to 1.
        states_embeddings = []  
        states_embeddings_elem_num = []  # record the number of elements in each state
        stage_masks = torch.zeros((1, actions.shape[1], self.max_stage_num), dtype=torch.float32, device=self.device)  # used to mask the inactive stages
        exec_masks = torch.zeros((1, actions.shape[1], self.max_exec_num), dtype=torch.float32, device=self.device)  # used to mask the inactive executors
        for seq_idx, state in enumerate(states[0]):  # since we force batch size to be 1, it is safe to directly fetch all states with index [0]
            dag_batch = state
            stage_to_job_map = dag_batch.batch
            stage_mask = dag_batch['stage_mask']

            dag_batch.to(self.device, non_blocking=True)

            # 2.1 compute node, dag, and global representations
            # h_dict is a dict of three features: node features, dag features, global features
            h_dict = self.state_encoder(dag_batch)

            # 2.2 generate stages (nodes) features
            # shape: (num_of_active_nodes, feature_size). note: num_of_active_nodes may change among states
            stage_features, stage_mask = self._generate_stage_features(dag_batch, h_dict)

            # 2.3 retrieve index of selected stage's job
            stage_idx = stage_indices[0, seq_idx]
            stage_idx_glob = pyg_utils.mask_to_index(stage_mask)[stage_idx]
            job_idx = stage_to_job_map[stage_idx_glob].item()

            # 2.4 generate executor features
            # shape: (1, feature_size)
            exec_features, exec_mask = self._generate_exec_features(dag_batch, h_dict, job_idx)
            
            # 2.5 embed each state features
            stage_embeddings = self.embed_stage_state(stage_features)
            exec_embeddings = self.embed_exec_state(exec_features)

            # 2.6 concat and add time embeddings
            total_embeddings = torch.cat((stage_embeddings, exec_embeddings), dim=0) + time_embeddings[0, seq_idx]
            states_embeddings.append(total_embeddings)
            states_embeddings_elem_num.append(total_embeddings.shape[0])
            stage_masks[0, seq_idx, stage_mask.sum():] = -INF  # set a very small value
            exec_masks[0, seq_idx, exec_mask.sum():] = -INF
        
        # Step 3: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = []
        action_embed_positions = []  # record the positions of action embeddings
        for i in range(len(states_embeddings)):
            stacked_input = torch.cat((returns_embeddings[0, i:i + 1], states_embeddings[i], action_embeddings[0, i:i + 1]), dim=0)
            stacked_inputs.append(stacked_input)
            action_embed_positions.append(states_embeddings_elem_num[i] + 2)
        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization
        
        # Step 4: feed stacked embeddings into the plm
        # 4.1 create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # Step 5: predict actions
        # 5.1 locate action, return embedding positions
        # we use action_embed_positions & return_embed_positions to locate the positions of action & return embedings
        # we will use them to locate other logits later
        action_embed_positions = np.cumsum(action_embed_positions)
        action_embed_positions = action_embed_positions[action_embed_positions < self.plm_embed_size] # avoid the error of out of bound
        return_embed_positions = np.ones_like(action_embed_positions)
        return_embed_positions[1:] = action_embed_positions[:-1] + 1

        # NOTE
        # each state features are composed of two parts:
        # node features: s_1, s_2, ..., s_n
        # exec features: s_(n+1)
        # each action is composed of two parts:
        # stage_idx: predicted with the node features
        # num_exec: predicted with the exec features
        # during training, logits will be predicted instead of explicitly producing stage_idx and num_exec

        # 5.1 predict next stage action with self.stage_action_head1
        stage_pred1 = None
        if use_head == UseStageHead.HEAD1 or use_head == UseStageHead.BOTH:
            stage_pred1 = []
            for i in range(len(action_embed_positions)):
                # for this head, we need to locate the logits corresponding to all node features (i.e., s_1, s_2, ..., s_n)
                # simply using `return_embed_positions[i]:action_embed_positions[i] - 2` will do.
                logits_used = logits[:, return_embed_positions[i]:action_embed_positions[i] - 2]
                stage_pred1.append(self.stage_action_head1(logits_used).reshape(-1))

        # 5.2 predict next stage action with self.stage_action_head2
        # for this head, we need to locate the logits corresponding to the last node features (i.e., s_n)
        # simply using `action_embed_positions[i] - 3` will do.
        stage_pred2 = None
        if use_head == UseStageHead.HEAD2 or use_head == UseStageHead.BOTH:
            logits_used = logits[:, action_embed_positions - 3]
            stage_pred2 = self.stage_action_head2(logits_used)
            stage_pred2 = stage_pred2 + stage_masks[:, :stage_pred2.shape[1], :]  # mask inactive stages

        # 5.3 predict the parallelism limit
        # for this head, we need to locate the logits corresponding to the last node features (i.e., s_(n+1))
        # simply using `action_embed_positions[i] - 2` will do.
        logits_used = logits[:, action_embed_positions - 2] # 报错的地方
        exec_pred = self.exec_action_head(logits_used)
        exec_pred = exec_pred + exec_masks[:, :exec_pred.shape[1], :]  # mask inactive executors

        return stage_pred1, stage_pred2, exec_pred

    def sample(self, state, target_return, timestep, like_decima=False, **kwargs):
        """
        Sample action function, used for evaluation/testing.
        """
        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.states_dq)):
            prev_return_embeddings = self.returns_dq[i]
            prev_state_embeddings = self.states_dq[i]
            prev_action_embeddings = self.actions_dq[i]
            prev_stacked_inputs.append(torch.cat((prev_return_embeddings, prev_state_embeddings, prev_action_embeddings), dim=1))
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int32, device=self.device).reshape(1, 1)

        return_embeddings = self.embed_return(target_return)
        time_embeddings = self.embed_timestep(timestep)

        return_embeddings = return_embeddings + time_embeddings

        # Step 3: predict the next stage to run
        dag_batch = state
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch['stage_mask']
        dag_batch.to(self.device, non_blocking=True)

        # 3.1 compute node, dag, and global representations
        # h_dict is a dict of three features: node features, dag features, global features
        h_dict = self.state_encoder(dag_batch)

        # 3.2 generate stages (nodes) features
        # shape: (num_of_active_nodes, feature_size). note: num_of_active_nodes may change among states
        stage_features, stage_mask = self._generate_stage_features(dag_batch, h_dict)
        stage_embeddings = self.embed_stage_state(stage_features).unsqueeze(0)
        stage_embeddings = stage_embeddings + time_embeddings

        # 3.3 stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, stage_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # 3.4 predict the next stage to run according to the specified manner
        if like_decima:
            logits_used = logits[:, -stage_embeddings.shape[1]:]
            stage_pred = self.stage_action_head1(logits_used).reshape(-1)
        else:
            logits_used = logits[:, -1:]
            stage_pred = self.stage_action_head2(logits_used)
            stage_pred = stage_pred.reshape(-1)
            stage_pred = stage_pred[:stage_mask.sum()]  # truncated inactive stages
        stage_idx, _ = self._sample(stage_pred)

        # Step 4: predict the parallelism limit
        # 4.1 retrieve index of selected stage's job
        stage_idx_glob = pyg_utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob].item()

        # 4.2 generate executor features
        # shape: (1, feature_size)
        exec_features, exec_mask = self._generate_exec_features(dag_batch, h_dict, job_idx)
        exec_embeddings = self.embed_exec_state(exec_features).unsqueeze(0)
        exec_embeddings = exec_embeddings + time_embeddings

        # 4.3 stack exec and previous embedding
        stacked_inputs = torch.cat((stacked_inputs, exec_embeddings), dim=1)  # mind the order
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]  # truncate sequence length (should not exceed plm embed size)
        stacked_inputs_ln = self.embed_ln(stacked_inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((stacked_inputs_ln.shape[0], stacked_inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
            stop_layer_idx=self.which_layer,
        )
        logits = transformer_outputs['last_hidden_state']
        if self.residual:
            logits = logits + stacked_inputs_ln  # residual add

        # 4.4 predict the parallelism limit
        logits_used = logits[:, -1:]
        exec_pred = self.exec_action_head(logits_used)
        exec_pred = exec_pred.reshape(-1)
        exec_pred = exec_pred[:exec_mask.sum()]  # truncated inactive executors
        num_exec, _ = self._sample(exec_pred)

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 2, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = stage_idx / self.max_stage_num
        action_tensor[..., 1] = num_exec / self.max_exec_num
        action_embeddings = self.embed_action(action_tensor)
        
        # update deques
        self.returns_dq.append(return_embeddings)
        self.states_dq.append(torch.cat((stage_embeddings, exec_embeddings), dim=1))  # mind the order
        self.actions_dq.append(action_embeddings)

        action = {
            'stage_idx': stage_idx,
            'job_idx': job_idx,
            'num_exec': num_exec
        }
        return action
    
    def clear_dq(self):
        self.states_dq.clear()
        self.actions_dq.clear()
        self.returns_dq.clear()
        
        self.states_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.actions_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))
        self.returns_dq.append(torch.zeros((1, 0, self.plm_embed_size), device=self.device))

    def _generate_stage_features(self, dag_batch, h_dict):
        stage_mask = dag_batch['stage_mask']  

        x = dag_batch.x[stage_mask]

        h_node = h_dict['node'][stage_mask]

        batch_masked = dag_batch.batch[stage_mask]
        h_dag_rpt = h_dict['dag'][batch_masked]

        # comments of wuduo: number of active stages
        num_stage_acts = stage_mask.sum() 

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0)

        # residual connections to original features
        stage_features = torch.cat(
            [
                x, 
                h_node, 
                h_dag_rpt, 
                h_glob_rpt
            ], 
            dim=1
        )
        return stage_features, stage_mask

    def _generate_exec_features(self, dag_batch, h_dict, job_indices):
        assert isinstance(job_indices, int) or isinstance(job_indices, list) and len(job_indices) == 1
        exec_mask = dag_batch['exec_mask']  # whether each exectutor can be allocated to the job, its sum is the number of idle executors that can be allocated

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, :3]  # 3 is the number of dag features
        x_dag = x_dag[job_indices]

        h_dag = h_dict['dag'][job_indices]
        h_glob = h_dict['glob']

        exec_mask = exec_mask[job_indices]

        x_dag = x_dag.unsqueeze(0)
        h_dag = h_dag.unsqueeze(0)
        exec_mask = exec_mask

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag, h_glob], dim=1)
        
        return x_h_dag, exec_mask

    def _sample(self, logits):
        pi = F.softmax(logits, 0).cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
