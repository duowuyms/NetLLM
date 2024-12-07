import random
import numpy as np
import torch


def process_batch_actions(batch_actions, max_stage_num, max_exec_num, device='cpu'):
    """
    Process a batch of actions sampled from the ExperienceDataset.
    """
    batch_actions_tensors = []
    batch_stage_indices, batch_num_execs = [], []
    for actions in batch_actions:
        actions_tensors = []
        stage_indices, num_execs = [], []
        for action in actions:
            stage_idx = action['stage_idx']
            num_exec = action['num_exec']
            action_tensor = torch.as_tensor([stage_idx / max_stage_num, num_exec / max_exec_num], dtype=torch.float32, device=device)
            actions_tensors.append(action_tensor)
            stage_indices.append(stage_idx)
            num_execs.append(num_exec)
        actions_tensors = torch.stack(actions_tensors, dim=0)
        batch_actions_tensors.append(actions_tensors)
        batch_stage_indices.append(torch.as_tensor(stage_indices, dtype=torch.long, device=device))
        batch_num_execs.append(torch.as_tensor(num_execs, dtype=torch.long, device=device))
    batch_actions_tensors = torch.stack(batch_actions_tensors, dim=0)
    batch_stage_indices = torch.stack(batch_stage_indices, dim=0)
    batch_num_execs = torch.stack(batch_num_execs, dim=0)
    return batch_actions_tensors, batch_stage_indices, batch_num_execs


def process_batch_rewards(batch_rewards, max_reward, min_reward, device='cpu'):
    """
    Process a batch of rewards sampled from the ExperienceDataset.
    """
    batch_rewards_tensors = torch.as_tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(2)
    batch_rewards_tensors = (batch_rewards_tensors - min_reward) / (max_reward - min_reward)  # normalized
    return batch_rewards_tensors


def process_batch_returns(batch_returns, device='cpu'):
    """
    Process a batch of returns sampled from the ExperienceDataset.
    """
    batch_returns_tensors = torch.as_tensor(batch_returns, dtype=torch.float32, device=device).unsqueeze(2)
    return batch_returns_tensors


def process_batch_timesteps(batch_timesteps, device='cpu'):
    """
    Process a batch of timesteps sampled from the ExperienceDataset.
    """
    batch_timesteps_tensors = torch.as_tensor(batch_timesteps, dtype=torch.int32, device=device)
    return batch_timesteps_tensors


def process_actions(actions, max_stage_num, max_exec_num, device='cpu'):
    """
    Process actions sampled from the DataLoader (batch size = 1).
    """
    actions_tensors = []
    stage_indices, num_execs = [], []
    for action in actions:
        stage_idx = action['stage_idx']
        num_exec = action['num_exec']
        action_tensor = torch.as_tensor([stage_idx / max_stage_num, num_exec / max_exec_num], dtype=torch.float32, device=device)
        actions_tensors.append(action_tensor)
        stage_indices.append(stage_idx)
        num_execs.append(num_exec)
    actions_tensors = torch.stack(actions_tensors, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    stage_indices = torch.cat(stage_indices, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    num_execs = torch.cat(num_execs, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    return actions_tensors, stage_indices, num_execs


def process_rewards(rewards, max_reward, min_reward, device='cpu'):
    """
    Process rewards sampled from the DataLoader (batch size = 1).
    """
    rewards_tensors = torch.stack(rewards, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    rewards_tensors = (rewards_tensors - min_reward) / (max_reward - min_reward)  # normalized
    return rewards_tensors


def process_returns(returns, max_return, min_return, device='cpu'):
    """
    Process returns sampled from the DataLoader (batch size = 1).
    """
    returns_tensors = torch.stack(returns, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    returns_tensors = (returns_tensors - min_return) / (max_return - min_return)  # normalized
    return returns_tensors


def process_timesteps(timesteps, device='cpu'):
    """
    Process timesteps sampled from the DataLoader (batch size = 1).
    """
    timesteps_tensors = torch.cat(timesteps, dim=0).unsqueeze(0).to(device=device, non_blocking=True)
    return timesteps_tensors


def set_random_seed(seed):
    """
    Set random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
