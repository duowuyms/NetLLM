import copy
import torch
import time

from baseline_special.env import Environment
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, MAX_VIDEO_BIT_RATE, DEFAULT_QUALITY
)
from plm_special.utils.utils import set_random_seed


def evaluate_on_env(args, env_settings, model, target_return, max_ep_num=50, process_reward_fn=None, seed=0):
    if process_reward_fn is None:
        process_reward_fn = lambda x: x

    eval_log = {}
    eval_start = time.time()

    with torch.no_grad():
        env = Environment(**env_settings)
        
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = torch.zeros((1, 1, S_INFO, S_LEN), dtype=torch.float32, device=args.device)
        timestep = 0
        target_return_clone = copy.deepcopy(target_return)
        ep_count = 0
        episodes_return, episodes_len = 0, 0
        
        set_random_seed(args.seed)

        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = env.get_video_chunk(bit_rate)

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            
            last_bit_rate = bit_rate

            # dequeue history record
            state = torch.roll(state, -1, dims=-1)

            # this should be S_INFO number of terms
            state[..., 0, -1] = VIDEO_BIT_RATE[bit_rate] / MAX_VIDEO_BIT_RATE # last quality
            state[..., 1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[..., 2, -1] = video_chunk_size / delay / M_IN_K  # kilo byte / ms
            state[..., 3, -1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[..., 4, :BITRATE_LEVELS] = torch.as_tensor(next_video_chunk_sizes, device=args.device, dtype=torch.float32) / M_IN_K / M_IN_K  # mega byte
            state[..., 5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

            if timestep > 0:  # skip the first reward like pensieve
                reward = process_reward_fn(reward)
                target_return = target_return - reward
                episodes_return += reward
                episodes_len += 1

            bit_rate = model.sample(state, target_return, timestep)
            timestep += 1

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                torch.zero_(state)
                timestep = 0
                target_return = copy.deepcopy(target_return_clone)

                ep_count += 1
                if ep_count >= max_ep_num:
                    break
    
    eval_log.update({
        'time/evaluation': time.time() - eval_start,
        'episodes_return': episodes_return,
        'episodes_len': episodes_len,
    })
    return eval_log
