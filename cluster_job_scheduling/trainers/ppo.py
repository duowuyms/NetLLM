from itertools import chain

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .trainer import Trainer
from spark_sched_sim import graph_utils


EPS = 1e-8


class RolloutDataset(Dataset):
    def __init__(self, obsns, acts, advgs, lgprobs):
        self.obsns = obsns
        self.acts = acts
        self.advgs = advgs
        self.lgprobs = lgprobs

    def __len__(self):
        return len(self.obsns)
    
    def __getitem__(self, idx):
        return self.obsns[idx], self.acts[idx], \
               self.advgs[idx], self.lgprobs[idx]
    

def collate_fn(batch):
    obsns, acts, advgs, lgprobs = zip(*batch)
    obsns = graph_utils.collate_obsns(obsns)
    acts = torch.stack(acts)
    advgs = torch.stack(advgs)
    lgprobs = torch.stack(lgprobs)
    return obsns, acts, advgs, lgprobs



class PPO(Trainer):
    '''Proximal Policy Optimization'''

    def __init__(
        self,
        agent_cfg,
        env_cfg,
        train_cfg
    ):  
        super().__init__(
            agent_cfg,
            env_cfg,
            train_cfg
        )

        self.entropy_coeff = train_cfg.get('entropy_coeff', 0.)
        self.clip_range = train_cfg.get('clip_range', .2)
        self.target_kl = train_cfg.get('target_kl', .01)
        self.num_epochs = train_cfg.get('num_epochs', 10)
        self.num_batches = train_cfg.get('num_batches', 3)


    def train_on_rollouts(self, rollout_buffers):
        obsns_list, actions_list, returns_list, baselines_list, lgprobs_list = \
            self._preprocess_rollouts(rollout_buffers)

        returns = np.array(list(chain(*returns_list)))
        baselines = np.concatenate(baselines_list)

        dataset = RolloutDataset(
            obsns = list(chain(*obsns_list)),
            acts = torch.tensor(list(chain(*actions_list))),
            advgs = torch.from_numpy(returns - baselines).float(),
            lgprobs = torch.tensor(list(chain(*lgprobs_list))))
        
        dataloader = DataLoader(
            dataset,
            batch_size = len(dataset) // self.num_batches + 1,
            shuffle = True,
            collate_fn = collate_fn)

        return self._train(dataloader)
    

    def _train(self, dataloader):
        policy_losses = []
        entropy_losses = []
        approx_kl_divs = []
        continue_training = True

        for _ in range(self.num_epochs):
            if not continue_training:
                break

            for obsns, actions, advgs, old_lgprobs in dataloader:
                total_loss, policy_loss, entropy_loss, approx_kl_div = \
                    self._compute_loss(obsns, actions, advgs, old_lgprobs)

                policy_losses += [policy_loss]
                entropy_losses += [entropy_loss]
                approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None \
                    and approx_kl_div > 1.5 * self.target_kl:
                    print(f'Early stopping due to reaching max kl: '
                            f'{approx_kl_div:.3f}')
                    continue_training = False
                    break

                self.agent.update_parameters(total_loss)
        
        return {
            'policy loss': np.abs(np.mean(policy_losses)),
            'entropy': np.abs(np.mean(entropy_losses)),
            'approx kl div': np.abs(np.mean(approx_kl_divs))
        }
    

    def _compute_loss(self, obsns, acts, advgs, old_lgprobs):
        '''CLIP loss'''
        lgprobs, entropies = self.agent.evaluate_actions(obsns, acts)

        advgs = (advgs - advgs.mean()) / (advgs.std() + EPS)
        log_ratio = lgprobs - old_lgprobs
        ratio = log_ratio.exp()

        policy_loss1 = advgs * ratio
        policy_loss2 = advgs * torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

        entropy_loss = -entropies.mean()

        loss = policy_loss + self.entropy_coeff * entropy_loss

        with torch.no_grad():
            approx_kl_div = ((ratio - 1) - log_ratio).mean().item()

        return loss, policy_loss.item(), entropy_loss.item(), approx_kl_div