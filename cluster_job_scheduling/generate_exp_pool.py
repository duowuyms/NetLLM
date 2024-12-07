import os
import pickle
import gymnasium as gym
import pathlib
import warnings

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint

from cfg_loader import load
from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from plm_special.data.exp_pool import ExperiencePool


def collect_experience(env_settings, scheduler, exp_pool_size, seed, complete_episode=False):
    env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_settings)
    if isinstance(scheduler, NeuralScheduler):
        env = NeuralActWrapper(env)
        env = scheduler.obs_wrapper_cls(env)

    obs, _ = env.reset(seed=seed, options=None)
    
    states, actions, rewards, dones, infos = [], [], [], [], []
    exp_pool_num = 0
    while True:
        if isinstance(scheduler, NeuralScheduler):
            action, *_ = scheduler(obs)
        else:
            action = scheduler(obs)
        next_obs, reward, done, _, info = env.step(action)
        states.append(obs), actions.append(action), rewards.append(reward), dones.append(done), infos.append(info)
        obs = next_obs
        if done:
            seed += 1  # increase seed to change another seed
            obs, _ = env.reset(seed=seed, options=None)
        exp_pool_num += 1
        if exp_pool_num >= exp_pool_size:
            # collection will terminate if:
            # a) complete_episode=True and done=True: force to complete the episode and now the episode is done
            # b) complete_episode=False: exit as long as reaching exp_pool_size (no need to complete the undone episode)
            if complete_episode and done or not complete_episode:
                break
    return states, actions, rewards, dones, infos


def run(args):
    # filter the annoying warning messages
    warnings.filterwarnings("ignore", category=UserWarning, message='.*get variables from other wrappers is deprecated and will be removed in v1.0.*')
    warnings.filterwarnings("ignore", category=UserWarning, message='.*is not within the observation space.*')

    env_settings = {
        'num_executors': args.num_executors,
        'job_arrival_cap': args.job_arrival_cap,
        'job_arrival_rate': args.job_arrival_rate,
        'moving_delay': args.moving_delay,
        'warmup_delay': args.warmup_delay,
        'dataset': args.dataset,
        'render_mode': args.render_mode,
        'warn': False,
    }

    exp_pool_dir = os.path.join('artifacts', 'exp_pool', args.dataset, f'exe_{args.num_executors}_cap_{args.job_arrival_cap}_rate_{args.job_arrival_rate}_md_{args.moving_delay}_'\
                               f'wd_{args.warmup_delay}_seed_{args.seed}', '_'.join(args.scheds))
    if not os.path.exists(exp_pool_dir):
        os.makedirs(exp_pool_dir)
    
    print('Environment settings:')
    pprint(env_settings)

    # create schedulers
    schedulers = []
    if 'fair' in args.scheds:
        fair_scheduler = RoundRobinScheduler(env_settings['num_executors'], dynamic_partition=True)
        schedulers.append(fair_scheduler)
    if 'fifo' in args.scheds:
        # set dynamic_partition=False, then RoundRobinScheduler (fair) turns to FIFOScheduler
        fifo_scheduler = RoundRobinScheduler(env_settings['num_executors'], dynamic_partition=False)  
        schedulers.append(fifo_scheduler)
    if 'decima' in args.scheds:
        cfg = load(os.path.join('config', 'decima_tpch.yaml'))
        agent_cfg = cfg['agent']
        agent_cfg.update({'num_executors': env_settings['num_executors'],
                          'state_dict_path': os.path.join('models', 'decima', 'model.pt')})  # load pretrained model for decima
        agent_cfg['device'] = args.device
        decima_scheduler = make_scheduler(agent_cfg)
        schedulers.append(decima_scheduler)
    assert len(schedulers) > 0, 'Please specify at least one valid scheduler (e.g., decima).'

    if len(schedulers) > 1:
        args.complete_episode = True  # if we use more than one schedulers to collect experience, we need to set complete_episode=True to aviod possible errors.
        print('Detect more than one schedulers, automatically set complete_episode to True.')
    
    print(f'Collect experience with scheduler(s) {", ".join(args.scheds)}...')
    states, actions, rewards, dones, infos = [], [], [], [], []
    for scheduler in schedulers:
        partial_states, partial_actions, partial_rewards, partial_dones, partial_infos = collect_experience(env_settings, scheduler, args.pool_size // len(schedulers), args.seed, args.complete_episode)
        states.extend(partial_states), actions.extend(partial_actions), rewards.extend(partial_rewards)
        dones.extend(partial_dones), infos.extend(partial_infos)

    exp_pool_path = os.path.join(exp_pool_dir, f'exp_pool_size_{args.pool_size}.pkl')
    exp_pool = ExperiencePool()
    for i in range(len(states)):
        exp_pool.add(state=states[i], action=actions[i], next_state=states[(i + 1) % len(states)], reward=rewards[i], done=dones[i], info=infos[i])
    pickle.dump(exp_pool, open(exp_pool_path, 'wb'))
    print(f'Done! Experience pool with actual size {len(states)} saved at:', exp_pool_path)


if __name__ == '__main__':
    # save final rendering to artifacts dir
    pathlib.Path('artifacts').mkdir(parents=True, exist_ok=True) 

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # experience pool generation settings
    parser.add_argument('--scheds', dest='scheds', nargs='*', help='which scheduler to run', default=['decima'])
    parser.add_argument('--pool-size', dest='pool_size', help='the size of the generated experience pool', type=int, default=1000)
    parser.add_argument('--complete-episode', dest='complete_episode', help='when collecting experience, the episode may not be done when we already collect enough experience, '\
                        'i.e., the experience pool size reaches pool_size. we can continue the episode until it is done by setting --complete-episode, and of cource, we are likely' \
                         ' to collect more experience than we expect in that case.', action='store_true')
    # environment settings
    parser.add_argument('--num-executors', dest='num_executors', help='the total number of executors in the simulation', type=int, default=50)
    parser.add_argument('--job-arrival-cap', dest='job_arrival_cap', help='the total number of jobs that arrive throughout the simulation', type=int, default=200)
    parser.add_argument('--job-arrival-rate', dest='job_arrival_rate', help='non-negative number that controls how quickly new jobs arrive into the system.', type=float, default=4.e-5)
    parser.add_argument('--moving-delay', dest='moving_delay', help='time in ms it takes for a executor to move between jobs', type=float, default=2000.)
    parser.add_argument('--warmup-delay', dest='warmup_delay', help='an executor is slower on its first task from  a stage if it was previously '\
                        'idle or moving jobs, which is caputred by adding a warmup delay to the task duration', default=1000.)
    parser.add_argument('--render-mode', dest='render_mode', help='if set to "human", then a visualization of the simulation is rendred in real time', default=None)
    parser.add_argument('--dataset', dest='dataset', help='choice of dataset to generate jobs from. Currently, only "tpch" is supported', default='tpch')
    # other settings
    parser.add_argument('--seed', dest='seed', help='Random seed', type=int, default=1)
    parser.add_argument('--device', dest='device', help='Device', default='cuda:0')
    args = parser.parse_args()
    run(args)
