import copy
import torch
import time

from spark_sched_sim.graph_utils import obs_to_pyg
from plm_special.models import UseStageHead
from plm_special.utils.utils import set_random_seed


def _evaluate_episode(args, env, model, target_return, max_ep_len, process_reward_fn=None, like_decima=False, seed=0):
    # evaluate episode using stage_action_head1 (predicting stage to run like decima)
    state, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False
    timestep = 0
    episode_return, episode_len = 0, 0
    
    model.eval()
    model.clear_dq()

    set_random_seed(args.seed)
    with torch.no_grad():
        for _ in range(max_ep_len):
            state = obs_to_pyg(state)
            action = model.sample(state, target_return, timestep, like_decima=like_decima)
            state, reward, terminated, truncated, _ = env.step(action)
            reward = process_reward_fn(reward)
            target_return -= reward
            timestep = min(timestep + 1, max_ep_len - 1)

            episode_return += reward
            episode_len += 1

            if terminated:
                break

    return episode_return, episode_len


def evaluate_episode(args, env, model, target_return, max_ep_len, process_reward_fn=None, use_head=UseStageHead.HEAD2, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    eval_log = {'ep_return_max': 0., 'ep_avg_return_max': 0.}
    eval_start = time.time()

    if use_head == UseStageHead.HEAD1 or use_head == UseStageHead.BOTH:
        # evaluate episode using stage_action_head1 (predicting stage to run like decima)
        target_return_clone = copy.deepcopy(target_return)
        episode_return1, episode_len1 = _evaluate_episode(args, env, model, target_return_clone, max_ep_len, process_reward_fn, like_decima=True, seed=seed)
        eval_log.update({            
            'ep_return/stage_head1': episode_return1,
            'ep_len/stage_head1': episode_len1,
            'ep_avg_return/stage_head1': episode_return1 / episode_len1,
        })
        eval_log['ep_return_max'] = max(eval_log['ep_return_max'], episode_return1)
        eval_log['ep_avg_return_max'] = max(eval_log['ep_avg_return_max'], episode_return1 / episode_len1)
    if use_head == UseStageHead.HEAD2 or use_head == UseStageHead.BOTH:
        # evaluate episode using stage_action_head2 (predicting stage in our own way)
        target_return_clone = copy.deepcopy(target_return)
        episode_return2, episode_len2 = _evaluate_episode(args, env, model, target_return_clone, max_ep_len, process_reward_fn, like_decima=False, seed=seed)
        eval_log.update({            
            'ep_return/stage_head2': episode_return2,
            'ep_len/stage_head2': episode_len2,
            'ep_avg_return/stage_head2': episode_return2 / episode_len2,
        })
        eval_log['ep_return_max'] = max(eval_log['ep_return_max'], episode_return2)
        eval_log['ep_avg_return_max'] = max(eval_log['ep_avg_return_max'], episode_return2 / episode_len2)
    
    eval_log.update({'time/evaluation': time.time() - eval_start})
    return eval_log
