"""
Run baselines such as Decima, FIFO, Fair Scheduling.
We do not provide training procedure here, as some of baselines are heuristic algorithms and there is already
a checkpoint model for Decima.
"""

'''Examples of how to run job scheduling simulations with different schedulers
'''
import os
import pickle
import time
import gymnasium as gym
import pathlib
import warnings


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint

from cfg_loader import load
from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from spark_sched_sim import metrics
from plm_special.utils import set_random_seed


def test(env_settings, scheduler, seed, result_path):
    env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_settings)
    if isinstance(scheduler, NeuralScheduler):
        env = NeuralActWrapper(env)
        env = scheduler.obs_wrapper_cls(env)

    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False

    set_random_seed(args.seed)
    
    while not (terminated or truncated):
        if isinstance(scheduler, NeuralScheduler):
            action, *_ = scheduler(obs)
        else:
            action = scheduler(obs)
        obs, _, terminated, truncated, _ = env.step(action)

    job_durations = metrics.job_durations(env) 
    job_durations = np.array(job_durations) * 1e-3  # millisecond -> second
    avg_job_duration = metrics.avg_job_duration(jd=job_durations)

    # cleanup rendering
    env.close()

    print(f'Done! Average job duration: {avg_job_duration:.1f} sec', flush=True)
    pickle.dump({'job_durations': job_durations, 'avg_job_duration': avg_job_duration}, open(result_path, 'wb'))
    print('Result saved at:', result_path)


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

    results_dir = os.path.join('artifacts', 'results', args.dataset, f'exe_{args.num_executors}_cap_{args.job_arrival_cap}_rate_{args.job_arrival_rate}_md_{args.moving_delay}_wd_{args.warmup_delay}_env_seed_{args.env_seed}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print('Environment settings:')
    pprint(env_settings)

    # create scheduler
    if args.sched == 'fair':
        scheduler = RoundRobinScheduler(env_settings['num_executors'], dynamic_partition=True)
    elif args.sched == 'fifo':
        # set dynamic_partition=False, then RoundRobinScheduler (fair) turns to FIFOScheduler
        scheduler = RoundRobinScheduler(env_settings['num_executors'], dynamic_partition=False)  
    elif args.sched == 'random':
        scheduler = RandomScheduler(seed=args.seed)
    elif args.sched == 'decima':
        cfg = load(os.path.join('cluster_job_scheduling_collated', 'config', 'decima_tpch.yaml'))
        agent_cfg = cfg['agent']
        agent_cfg.update({'num_executors': env_settings['num_executors']})
        if args.state_dict_path is not None:
            agent_cfg.update({'state_dict_path': args.state_dict_path})   # load pretrained model for decima
        agent_cfg['device'] = args.device
        # scheduler = DecimaScheduler(device=args.device, **agent_cfg)
        scheduler = make_scheduler(agent_cfg)
    else:
        raise ValueError(f'No scheduler called {args.sched}')
    
    print(f'Testing scheduler {args.sched}...')
    result_path = os.path.join(results_dir, f'result_{args.sched}_seed_{args.seed}_{args.result_file_label}.pkl')
    start_time = time.time()
    test(env_settings, scheduler, args.env_seed, result_path)
    print('Test time:', time.time() - start_time)


if __name__ == '__main__':
    # save final rendering to artifacts dir
    pathlib.Path('artifacts').mkdir(parents=True, exist_ok=True) 

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # scheduling algorithms
    # decima is neural-based RL model, while fair, fifo and random are heuristic algorithms
    parser.add_argument('--sched', choices=['fair', 'fifo', 'random', 'decima'], dest='sched', help='which scheduler to run', default='decima')
    # environment settings
    parser.add_argument('--num-executors', dest='num_executors', help='the total number of executors in the simulation', type=int, default=50)
    parser.add_argument('--job-arrival-cap', dest='job_arrival_cap', help='the total number of jobs that arrive throughout the simulation', type=int, default=200)
    parser.add_argument('--job-arrival-rate', dest='job_arrival_rate', help='non-negative number that controls how quickly new jobs arrive into the system.', type=float, default=4.e-5)
    parser.add_argument('--moving-delay', dest='moving_delay', help='time in ms it takes for a executor to move between jobs', type=float, default=2000.)
    parser.add_argument('--warmup-delay', dest='warmup_delay', help='an executor is slower on its first task from  a stage if it was previously '\
                        'idle or moving jobs, which is caputred by adding a warmup delay to the task duration', default=1000.)
    parser.add_argument('--render-mode', dest='render_mode', help='if set to "human", then a visualization of the simulation is rendred in real time', default=None)
    parser.add_argument('--dataset', dest='dataset', help='choice of dataset to generate jobs from. Currently, only "tpch" is supported', default='tpch')
    parser.add_argument('--state-dict-path', help='the path of decima state dict', default=None)
    # other settings
    parser.add_argument('--seed', dest='seed', help='Random seed', type=int, default=1)
    parser.add_argument('--env-seed', dest='env_seed', help='Environment random seed', type=int, default=None)
    parser.add_argument('--result-file-label', help='Label to distingush result file', type=str, default='pretrained')
    parser.add_argument('--device', dest='device', help='Device', default='cuda:7')
    args = parser.parse_args()

    if args.env_seed is None:
        args.env_seed = args.seed  # by default, env_seed is equal to random seed
        
    # command examples:
    # python run_baseline.py --sched decima --num-executors 50 --job-arrival-cap 200  --seed 666 --device cuda:0
    # python run_baseline.py --sched fair --num-executors 50 --job-arrival-cap 200  --seed 666 
    # python run_baseline.py --sched fifo --num-executors 50 --job-arrival-cap 200  --seed 666 

    run(args)
