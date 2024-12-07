import numpy as np
import torch
import torch.profiler

from .trainer import Trainer
from spark_sched_sim.graph_utils import collate_obsns




class VPG(Trainer):
    '''Vanilla Policy Gradient'''

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
    

    def train_on_rollouts(self, rollout_buffers):
        obsns_list, actions_list, returns_list, baselines_list, lgprobs_list = \
            self._preprocess_rollouts(rollout_buffers)

        policy_losses = []
        entropy_losses = []

        gen = zip(obsns_list, actions_list, returns_list, baselines_list, lgprobs_list)
        for obsns, actions, returns, baselines, old_lgprobs in gen:
            obsns = collate_obsns(obsns)
            actions = torch.tensor(actions)
            lgprobs, entropies = self.agent.evaluate_actions(obsns, actions)

            # re-computed log-probs don't exactly match the original ones,
            # but it doesn't seem to affect training
            # with torch.no_grad():
            #     diff = (lgprobs - torch.tensor(old_lgprobs)).abs()
            #     assert lgprobs.allclose(torch.tensor(old_lgprobs))

            adv = torch.from_numpy(returns - baselines).float()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -(lgprobs * adv).mean()
            policy_losses += [policy_loss.item()]

            entropy_loss = -entropies.mean()
            entropy_losses += [entropy_loss.item()]

            loss = policy_loss + self.entropy_coeff * entropy_loss
            loss.backward()

        self.agent.update_parameters()
        
        return {
            'policy loss': np.mean(policy_losses),
            'entropy': np.mean(entropy_losses)
        }