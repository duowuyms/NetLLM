import numpy as np
from torch.utils.data import Dataset


def discount_returns(rewards, gamma, scale):
    returns = [0 for _ in range(len(rewards))]
    returns[-1] = rewards[-1]
    for i in reversed(range(len(rewards) - 1)):
        returns[i] = rewards[i] + gamma * returns[i + 1]
    for i in range(len(returns)):
        returns[i] /= scale  # scale down return
    return returns


class ExperienceDataset(Dataset):
    """
    A dataset class that wraps the experience pool.
    """
    def __init__(self, exp_pool, gamma=1., scale=10, max_length=30, sample_step=None) -> None:
        """
        :param exp_pool: the experience pool
        :param gamma: the reward discounted factor
        :param scale: the factor to scale the return
        :param max_length: the w value in our paper, see the paper for details.
        """
        if sample_step is None:
            sample_step = max_length

        self.exp_pool = exp_pool
        self.exp_pool_size = len(exp_pool)
        self.gamma = gamma
        self.scale = scale
        self.max_length = max_length

        self.returns = []
        self.timesteps = []
        self.rewards = []

        self.exp_dataset_info = {}

        self._normalize_rewards()
        self._compute_returns()
        self.exp_dataset_info.update({
            'max_action': max(self.actions),
            'min_action': min(self.actions)
        })

        self.dataset_indices = list(range(0, self.exp_pool_size - max_length + 1, min(sample_step, max_length)))
    
    def sample_batch(self, batch_size=1, batch_indices=None):
        """
        Sample a batch of data from the experience pool.
        :param batch_size: the size of a batch. For CJS task, batch_size should be set to 1 due to the unstructural data format.
        """
        if batch_indices is None:
            batch_indices = np.random.choice(len(self.dataset_indices), size=batch_size)
        batch_states, batch_actions, batch_returns, batch_timesteps = [], [], [], []
        for i in range(batch_size):
            states, actions, returns, timesteps = self[batch_indices[i]]
            batch_states.append(states)
            batch_actions.append(actions)
            batch_returns.append(returns)
            batch_timesteps.append(timesteps)
        return batch_states, batch_actions, batch_returns, batch_timesteps
    
    @property
    def states(self):
        return self.exp_pool.states

    @property
    def actions(self):
        return self.exp_pool.actions
    
    @property
    def dones(self):
        return self.exp_pool.dones
    
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, index):
        start = self.dataset_indices[index]
        end = start + self.max_length
        return self.states[start:end], self.actions[start:end], self.returns[start:end], self.timesteps[start:end]

    def _normalize_rewards(self):
        min_reward, max_reward = min(self.exp_pool.rewards), max(self.exp_pool.rewards)
        rewards = (np.array(self.exp_pool.rewards) - min_reward) / (max_reward - min_reward)
        self.rewards = rewards.tolist()
        self.exp_dataset_info.update({
            'max_reward': max_reward,
            'min_reward': min_reward,
        })

    def _compute_returns(self):
        """
        Compute returns (discounted cumulative rewards)
        """
        episode_start = 0
        while episode_start < self.exp_pool_size:
            try:
                episode_end = self.dones.index(True, episode_start) + 1
            except ValueError:
                episode_end = self.exp_pool_size
            self.returns.extend(discount_returns(self.rewards[episode_start:episode_end], self.gamma, self.scale))
            self.timesteps += list(range(episode_end - episode_start))
            episode_start = episode_end
        assert len(self.returns) == len(self.timesteps)
        self.exp_dataset_info.update({
            # for normalizing rewards/returns
            'max_return': max(self.returns),
            'min_return': min(self.returns),

            # to help determine the maximum size of timesteps embedding
            'min_timestep': min(self.timesteps),
            'max_timestep': max(self.timesteps),
        })
