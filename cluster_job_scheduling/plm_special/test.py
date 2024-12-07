import copy
import numpy as np
import gymnasium as gym
import torch
import time

from spark_sched_sim import metrics
from spark_sched_sim.wrappers import NeuralActWrapper, DAGNNObsWrapper
from spark_sched_sim.graph_utils import obs_to_pyg
from plm_special.models import UseStageHead
from plm_special.utils.utils import set_random_seed


def _test_on_env(args, model, env_settings, target_return, max_ep_len, process_reward_fn=None, like_decima=False, seed=0):
    env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_settings)
    env = NeuralActWrapper(env)
    env = DAGNNObsWrapper(env)
    state, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False
    timestep = 0
    episode_len = 0
    
    model.eval()
    model.clear_dq()
    
    set_random_seed(args.seed)
    with torch.no_grad():
        while not (terminated or truncated):
            state = obs_to_pyg(state)
            action = model.sample(state, target_return, timestep, like_decima=like_decima)
            state, reward, terminated, truncated, _ = env.step(action)
            reward = process_reward_fn(reward)
            target_return -= reward
            timestep = min(timestep + 1, max_ep_len - 1)
            episode_len += 1

    job_durations = metrics.job_durations(env)
    job_durations = np.array(job_durations) * 1e-3  # millisecond -> second
    avg_job_duration = metrics.avg_job_duration(jd=job_durations)

    # cleanup rendering
    env.close()
    return env, avg_job_duration, episode_len, job_durations


def test_on_env(args, model, env_settings, target_return, max_ep_len, process_reward_fn=None, use_head=UseStageHead.HEAD2, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    test_log = {}
    test_start = time.time()

    if use_head == UseStageHead.HEAD1 or use_head == UseStageHead.BOTH:
        print('Test using stage_action_head1 (predicting stage to run like decima).')
        target_return_clone = copy.deepcopy(target_return)
        env1, avg_job_duration1, episode_len1, jd1 = _test_on_env(args, model, env_settings, target_return_clone, max_ep_len, process_reward_fn, like_decima=True, seed=seed)
        test_log.update({
            'avg_job_duration1': avg_job_duration1, 
            'episode_len1': episode_len1,
            'job_durations1': jd1,
        })
        print(f'Done! Average job duration: {avg_job_duration1:.1f} sec', flush=True)
    if use_head == UseStageHead.HEAD2 or use_head == UseStageHead.BOTH:
        print('Test using stage_action_head2 (predicting stage in our own way).')
        target_return_clone = copy.deepcopy(target_return)
        env2, avg_job_duration2, episode_len2, jd2 = _test_on_env(args, model, env_settings, target_return_clone, max_ep_len, process_reward_fn, like_decima=False, seed=seed)
        test_log.update({
            'avg_job_duration2': avg_job_duration2, 
            'episode_len2': episode_len2,
            'job_durations2': jd2,
        })
        print(f'Done! Average job duration: {avg_job_duration2:.1f} sec', flush=True)
    test_log.update({'time': time.time() - test_start})
    return test_log