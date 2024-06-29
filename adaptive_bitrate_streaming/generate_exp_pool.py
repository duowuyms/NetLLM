import argparse
import os
import pickle
import itertools
import random
import numpy as np
import tensorflow as tf
import baseline_special.a3c as a3c
import baseline_special.env as env
from numba import jit
from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import (
    REBUF_PENALTY, SMOOTH_PENALTY, DEFAULT_QUALITY, S_INFO, S_LEN, A_DIM, BITRATE_LEVELS, BUFFER_NORM_FACTOR,
    M_IN_K, SMOOTH_PENALTY, VIDEO_BIT_RATE, CHUNK_TIL_VIDEO_END_CAP, RAND_RANGE, DEFAULT_QUALITY, TOTAL_VIDEO_CHUNK
)
from plm_special.utils.utils import action2bitrate
from plm_special.data.exp_pool import ExperiencePool

PENSIEVE = 0
MPC = 1
BBA = 2


# =========================================================================
# ====================== Pensieve Special (Start) =========================
# =========================================================================

def pensieve(actor, state, last_bit_rate):
    action_prob = actor.predict(np.reshape(state, (1, S_INFO ,S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    # Note: we need to discretize the probability into 1 / RAND_RANGE steps,
    # because there is an intrinsic discrepancy in passing single state and batch states
    action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    bit_rate = action2bitrate(action, last_bit_rate)
    return bit_rate

# =========================================================================
# ======================= Pensieve Special (End) ==========================
# =========================================================================


# =========================================================================
# ========================= MPC Special (Start) ===========================
# =========================================================================

CHUNK_COMBO_OPTIONS = np.array([combo for combo in itertools.product(
                range(6), repeat=5)])
MPC_FUTURE_CHUNK_COUNT = 5


@jit(nopython=True)
def next_possible_bitrates(br):
    next_brs = [br - 1 ,br ,br + 1]
    next_brs = [a for a in next_brs if 0 <= a <= 5]
    return next_brs


@jit(nopython=True)
def calculate_jump_action_combo(br):
    all_combos = CHUNK_COMBO_OPTIONS
    combos = np.empty((0, 5), np.int64)
    #combos = np.expand_dims( combos ,axis=0 )
    for combo in all_combos:
        br1 = combo[0]
        if br1 in next_possible_bitrates( br ):
            br2 = combo[1]
            if br2 in next_possible_bitrates( br1 ):
                br3 = combo[2]
                if br3 in next_possible_bitrates( br2 ):
                    br4 = combo[3]
                    if br4 in next_possible_bitrates( br3 ):
                        br5 = combo[4]
                        if br5 in next_possible_bitrates( br4 ):
                            combo = np.expand_dims( combo ,axis=0 )
                            combos = np.append(combos, combo, axis=0)

    return combos


@jit(nopython=True)
def get_chunk_size(quality, index, size_video_array):
    if (index < 0 or index > TOTAL_VIDEO_CHUNK):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is
    # highest and this pertains to video1)
    return size_video_array[quality, index]


@jit(nopython=True)
def calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate, last_index, future_bandwidth, jump_action_combos, video_bit_rate):
    max_reward = -100000000
    start_buffer = buffer_size

    #jump_action_combos = calculate_jump_action_combo(bit_rate)
    for full_combo in jump_action_combos:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int( bit_rate )
        for position in range( 0, len( combo ) ):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            download_time = (get_chunk_size(chunk_quality, index, size_video_array) / 1000000.) / future_bandwidth
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += video_bit_rate[chunk_quality]
            smoothness_diffs += abs(
                video_bit_rate[chunk_quality] - video_bit_rate[last_quality] )
            last_quality = chunk_quality

        reward = (bitrate_sum / 1000.) - (REBUF_PENALTY *
                                          curr_rebuffer_time) - (smoothness_diffs / 1000.)
        if reward >= max_reward:
            best_combo = combo
            max_reward = reward
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]
    return send_data


def mpc(size_video_array, state, bit_rate, buffer_size, video_chunk_remain, video_bit_rate, past_errors, past_bandwidth_ests, combo_dict):
    curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    if (len(past_bandwidth_ests) > 0):
        curr_error = abs(past_bandwidth_ests[-1]- state[2, -1]) / float(state[2, -1])
    past_errors.append(curr_error)

    # pick bitrate according to MPC
    # first get harmonic mean of last 5 bandwidths
    past_bandwidths = state[2, -5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]
    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += (1 / float(past_val))
    harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if (len(past_errors) < 5):
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth/(1 + max_error)  # robustMPC here
    past_bandwidth_ests.append(harmonic_bandwidth)

    # future chunks length (try 4 if that many remaining)
    last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
    future_chunk_length = MPC_FUTURE_CHUNK_COUNT
    if (TOTAL_VIDEO_CHUNK - last_index < 5):
        future_chunk_length = TOTAL_VIDEO_CHUNK - last_index

    jump_action_combos = combo_dict[str(bit_rate)]

    bit_rate = calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate,
                                    last_index, future_bandwidth, jump_action_combos, video_bit_rate)
    return bit_rate

# =========================================================================
# ========================== MPC Special (End) ============================
# =========================================================================


# =========================================================================
# ========================= BBA Special (Start) ===========================
# =========================================================================

RESEVOIR = 5  # BBA
CUSHION = 10  # BBA


def bba(buffer_size):
    if buffer_size < RESEVOIR:
        bit_rate = 0
    elif buffer_size >= RESEVOIR + CUSHION:
        bit_rate = BITRATE_LEVELS - 1
    else:
        bit_rate = (BITRATE_LEVELS - 1) * (buffer_size - RESEVOIR) / float(CUSHION)
    bit_rate = int(bit_rate)
    return bit_rate

# =========================================================================
# ========================== BBA Special (End) ============================
# =========================================================================


def collect_experience(args, model, model_name, env_settings, trace_num, sess, actor=None):
    video_size_dir = cfg.video_size_dirs[args.video]
    net_env = env.Environment(**env_settings)

    total_states = []
    total_actions = []
    total_rewards = []
    total_dones = []

    print('Collect experience with model', model_name)
    # with tf.Session() as sess:
    if model == PENSIEVE:
        if actor is None:
            model_path = cfg.baseline_model_paths[model_name]
            actor = a3c.ActorNetwork(sess, state_dim=[S_INFO ,S_LEN] ,action_dim=A_DIM , bitrate_dim=BITRATE_LEVELS)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()  # save neural net parameters
            saver.restore(sess, model_path)  # restore neural net parameters
            print("Testing model restored.")
    elif model == MPC:
        combo_dict = {'0': calculate_jump_action_combo(0),
                        '1': calculate_jump_action_combo(1),
                        '2': calculate_jump_action_combo(2),
                        '3': calculate_jump_action_combo(3),
                        '4': calculate_jump_action_combo(4),
                        '5': calculate_jump_action_combo(5)}
        size_video_array =[]
        for bitrate in range(BITRATE_LEVELS):
            video_size = []
            with open(video_size_dir + 'video_size_' + str(bitrate)) as f:
                for line in f:
                    video_size.append( int( line.split()[0] ) )
            size_video_array.append(video_size)
        size_video_array = np.array(np.squeeze(size_video_array))
        assert len(VIDEO_BIT_RATE) == BITRATE_LEVELS
        video_bit_rate = np.array(VIDEO_BIT_RATE)
        past_errors = []
        past_bandwidth_ests = []

    states = []
    actions = []
    rewards = []
    dones = []
    all_rewards = []  # for debug use

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    test_trace_count = 0
    all_rewards  = []

    while True:  # serve video forever
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        states.append(state)
        actions.append(bit_rate)
        rewards.append(reward)
        dones.append(end_of_video)
        all_rewards.append(reward)  # for debug use

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                        float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
                        float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :BITRATE_LEVELS] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        if model == PENSIEVE:
            bit_rate = pensieve(actor, state, last_bit_rate)
        elif model == MPC:
            bit_rate = mpc(size_video_array, state, bit_rate, buffer_size, video_chunk_remain, video_bit_rate, past_errors,
                            past_bandwidth_ests, combo_dict)
        else:
            bit_rate = bba(buffer_size)

        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state = np.zeros((S_INFO, S_LEN), dtype=np.float32)

            total_states.extend(states[1:])
            total_actions.extend(actions[1:])
            total_rewards.extend(rewards[1:])
            total_dones.extend(dones[1:])

            states.clear()
            actions.clear()
            rewards.clear()
            dones.clear()

            test_trace_count += 1
            if test_trace_count >= trace_num:
                break
    print('Done!', np.mean(all_rewards))
    return total_states, total_actions, total_rewards, total_dones, actor


def run(args):
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)  
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
    np.random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    random.seed(args.seed)

    np.random.seed(args.seed)
    
    assert len(args.models) >= 1, 'Please specify at least one model for experience pool generation.'
    for model_name in args.models:
        assert model_name in ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real', 'mpc', 'bba'], f"Unknown model {model_name}"

    exp_pools_dir = os.path.join(cfg.exp_pools_dir, args.trace + f'_{args.video}', '_'.join(args.models), f'seed_{args.seed}_trace_num_{args.trace_num}_fixed_{args.fixed_order}')
    os.makedirs(exp_pools_dir, exist_ok=True)
    exp_pool = ExperiencePool()

    actor = None
    with tf.Session() as sess:
        trace_dir = cfg.trace_dirs[args.trace]
        video_size_dir = cfg.video_size_dirs[args.video]

        all_cooked_time ,all_cooked_bw ,all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
        trace_num = min(args.trace_num, len(all_file_names))
        if trace_num == -1:
            trace_num = len(all_file_names)
            args.fixed_order = True

        env_settings = {
            'all_cooked_time': all_cooked_time,
            'all_cooked_bw': all_cooked_bw,
            'all_file_names': all_file_names,
            'all_mahimahi_ptrs': all_mahimahi_ptrs,
            'video_size_dir': video_size_dir,
            'fixed': args.fixed_order,
            'trace_num': trace_num,
        }

        for model_name in args.models:
            if model_name in  ['genet', 'udr_1', 'udr_2', 'udr_3', 'udr_real']:
                model = PENSIEVE
            elif model_name == 'mpc':
                model = MPC
            else:
                model = BBA
            states, actions, rewards, dones, actor = collect_experience(args, model, model_name, env_settings, trace_num, sess, actor)
            for i in range(len(states)):
                exp_pool.add(state=states[i], action=actions[i], reward=rewards[i], done=dones[i])
    exp_pool_path = os.path.join(exp_pools_dir, 'exp_pool.pkl')
    pickle.dump(exp_pool, open(exp_pool_path, 'wb'))
    print(f"Done. Experience pool saved at:", exp_pool_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', help='choose one or more from [genet, udr_1, udr_2, udr_3, udr_real, mpc, bba]', nargs='*', default='genet')
    parser.add_argument("--trace", help='name of traces (e.g., fcc-train)')
    parser.add_argument('--video', help='name of videos (e.g., video1)')
    parser.add_argument('--trace-num', type=int, help='number of traces. if set to -1, use all traces in the trace dir.', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=100003)
    parser.add_argument('--cuda-id', type=int, help='cuda device idx', default=0)
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    args = parser.parse_args()

    # >>> debug <<<
    # args.models = ['bba', 'mpc', 'genet']
    # args.trace = 'fcc-train'
    # args.video = 'video1'
    # args.trace_num = -1
    # args.seed = 1
    # args.fixed_order = True
    # >>> debug <<<

    # command examples:
    # python generate_exp_pool.py --models genet --traces fcc-train --video video1 --trace-num -1 --seed 1 --fixed-order --cuda-id 0

    print(args)

    run(args)
