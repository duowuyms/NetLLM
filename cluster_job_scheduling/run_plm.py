import os
import sys
import numpy as np
import torch
import gymnasium as gym
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from cfg_loader import load
from spark_sched_sim.wrappers import DAGNNObsWrapper, NeuralActWrapper
from spark_sched_sim.schedulers import make_scheduler

from plm_special import Trainer, evaluate_episode, test_on_env
from plm_special.data import ExperienceDataset
from plm_special.models import OfflineRLPolicy, peft_model, EncoderNetwork, UseStageHead
from plm_special.utils import load_plm, ConsoleLogger, set_random_seed


PLM_TYPES = ['gpt2', 'llama', 't5-lm', 'llama_random']
PLM_SIZES = ['small', 'base', 'large', 'xl', 'xll', 'tiny']
PLM_DIR = '../../downloaded_plms' if 'cluster_job_scheduling' in os.getcwd() else '../downloaded_plms'
PLM_FT_DIR = 'artifacts/ft_plms'
RESULTS_DIR = 'artifacts/results/tpch' if 'cluster_job_scheduling' not in os.getcwd() else 'artifacts/results/tpch'
PLM_EMBED_SIZES = {
    'gpt2': {
        'base': 1024,
        'small': 768,
        'large': 1280,
        'xl': 1600,
    },
    'llama': {
        'base': 4096,
        'tiny': 2048,
    },
    'llama_random': {
        'base': 4096,
    },
    't5-lm': {
        'base': 768,
        'small': 512,
        'large': 4096,
        'xl': 2048,
    }
}
PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48,
    },
    'llama': {
        'base': 32,
        'tiny': 22,
    },
    'llama_random': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}

NUM_NODE_FEATURES = 5
NUM_DAG_FEATURES = 3
DEFAULT_USE_HEAD = UseStageHead.HEAD2


def save_model(args, model, save_dir):
    if args.rank > 0:
        # save low rank weights
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # low rank is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load low rank weights
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # low rank is disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def train(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )

    start_iter = 0
    if args.resume_dir is not None:
        model = load_model(args, model, args.resume_dir)
        start_iter = start_iter if args.resume_iter is None else args.resume_iter
        print('Resume model from:', args.resume_dir)
        print(f'Start training from Iter #{start_iter}')

    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps, use_head=args.use_head)
    eval_env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **eval_env_settings)
    eval_env = NeuralActWrapper(eval_env)
    eval_env = DAGNNObsWrapper(eval_env)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    max_ep_len = exp_dataset_info.max_timestep + 1
    best_eval_return = 0.

    total_train_losses = []
    for iter in range(start_iter, args.num_iters):
        print('='* 20, f'Training Iteration #{iter}', '=' * 20)
        train_logs, train_losses = trainer.train_iteration(args.num_steps_per_iter)
        total_train_losses.extend(train_losses)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        if iter % args.save_checkpoint_per_iter == 0:  # save checkpoint
            checkpoint_dir_iter = os.path.join(checkpoint_dir, str(iter))
            if not os.path.exists(checkpoint_dir_iter):
                os.makedirs(checkpoint_dir_iter)
            save_model(args, model, checkpoint_dir_iter)
            print('Checkpoint saved at:', checkpoint_dir_iter)

        if iter % args.eval_per_iter == 0:
            eval_logs = evaluate_episode(args, env=eval_env, model=model, target_return=target_return, max_ep_len=min(args.eval_max_ep_len, max_ep_len),
                                         process_reward_fn=eval_process_reward_fn, use_head=args.use_head, seed=args.env_seed)
            max_episode_return = eval_logs['ep_avg_return_max']
            if best_eval_return < max_episode_return:
                best_eval_return = max_episode_return
                save_model(args, model, best_model_dir)
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test(args, model, exp_dataset_info, env_settings, model_dir, result_path, test_process_reward_fn):
    model = load_model(args, model, model_dir)
    print('Load model from:', model_dir)
    max_ep_len = exp_dataset_info.max_timestep + 1
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, env_settings, target_return, max_ep_len, test_process_reward_fn, use_head=args.use_head, seed=args.env_seed)
    print(results)
    print('Test time:', results['time'])
    pickle.dump(results, open(result_path, 'wb'))
    print('Results saved at:', result_path)


def run(args):
    assert args.plm_type in PLM_TYPES
    assert args.plm_size in PLM_SIZES
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    
    if args.use_head == 1:
        args.use_head = UseStageHead.BOTH
    elif args.use_head == 2:
        args.use_head = UseStageHead.HEAD1
    elif args.use_head == 3:
        args.use_head = UseStageHead.HEAD2
    else:
        raise ValueError(f'No such head as {args.use_head}')

    # 1. set seed
    set_random_seed(args.seed)

    # 2. extract training experience pool information
    train_exp_pool_info = args.exp_pool_path.split('/')[-3:]
    train_exp_pool_info[-1] = train_exp_pool_info[-1][:-4]
    train_exp_pool_info = '_'.join(train_exp_pool_info)

    # 3. create environment setting
    env_info = f'exe_{args.num_executors}_cap_{args.job_arrival_cap}_rate_{args.job_arrival_rate}_md_{args.moving_delay}_wd_{args.warmup_delay}_env_seed_{args.env_seed}'
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

    # 4. create training dataset, fetch info
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.K, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 5. create model
    # 5.1 load plm
    # args.device_out and args.device_mid are used used for model parallelism (currently only necessary for llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    # plm, *_ = load_plm(args.plm_type, os.path.join(PLM_DIR, args.plm_type, args.plm_size), 
    #                    device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)
    plm, *_ = load_plm(args.plm_type, os.path.join(PLM_DIR, 'tinyllama'), 
                       device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)

    if args.plm_type != 'llama':
        plm = plm.to(args.device)
    
    if args.rank != -1:
        plm = peft_model(plm, args.plm_type, rank=args.rank)

    # 5.2 create state encoder
    if args.pt_encoder_config is not None:
        cfg = load(filename=args.pt_encoder_config)
        agent_cfg = cfg['agent']
        agent_cfg.update({'num_executors': 1})  # doesn't matter
        decima = make_scheduler(agent_cfg)
        state_encoder = decima.actor.encoder  # we only needs encoder
        args.state_feature_dim = agent_cfg['embed_dim']
    else:
        assert args.state_feature_dim is not None, 'please specify state feature dim if you dont use a pretrained encoder'
        state_encoder = EncoderNetwork(num_node_features=NUM_NODE_FEATURES, hidden_dim=args.state_feature_dim >> 1, embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)
    args.stage_embed_dim = NUM_NODE_FEATURES + args.state_feature_dim * 3 
    args.exec_embed_dim = NUM_DAG_FEATURES + args.state_feature_dim * 2

    # 5.3 create OfflineRLPolicy
    plm_embed_size = PLM_EMBED_SIZES[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(stage_state_dim=args.stage_embed_dim, exec_state_dim=args.exec_embed_dim, max_stage_num=args.max_stage_num,
                                               max_exec_num=args.max_exec_num, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size, 
                                               max_length=args.K, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out,
                                               which_layer=args.which_layer)

    # 6. handling directory and path
    models_dir = os.path.join(PLM_FT_DIR, f'{args.plm_type}_{args.plm_size}', train_exp_pool_info + f'_ss_{args.sample_step}', f'peft_{args.rank}_K_{args.K}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'\
                              f'_exec_{args.max_exec_num}_stage_{args.max_stage_num}_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_iters_{args.num_iters}_steps_{args.num_steps_per_iter}_seed_{args.seed}')
    results_dir = os.path.join(RESULTS_DIR, env_info)
    checkpoint_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_checkpoint')
    best_model_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_best_model')
    result_path = os.path.join(results_dir, f'early_stop_{args.which_layer}_results_dt_{args.plm_type}_{args.plm_size}_peft_{args.rank}_K_{args.K}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}.pkl')

    # 7. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.train:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        console_log = open(os.path.join(models_dir, f'early_stop_{args.which_layer}_console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        train(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, process_reward)
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.model_dir if args.model_dir is not None else best_model_dir
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, rl_policy, exp_dataset_info, env_settings, model_dir, result_path, process_reward)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument('--exp-pool-path', help='the path storing the experience pool file for training', 
                        default='artifacts/exp_pool/exp_pool.pkl')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    parser.add_argument('--num-executors', help='the total number of executors in the simulation', type=int, default=50)
    parser.add_argument('--job-arrival-cap', help='the total number of jobs that arrive throughout the simulation', type=int, default=200)
    parser.add_argument('--job-arrival-rate', help='non-negative number that controls how quickly new jobs arrive into the system.', type=float, default=4.e-5)
    parser.add_argument('--moving-delay', help='time in ms it takes for a executor to move between jobs', type=float, default=2000.)
    parser.add_argument('--warmup-delay', help='an executor is slower on its first task from  a stage if it was previously '\
                        'idle or moving jobs, which is caputred by adding a warmup delay to the task duration', type=float, default=1000.)
    parser.add_argument('--render-mode', dest='render_mode', help='if set to "human", then a visualization of the simulation is rendred in real time', default=None)
    parser.add_argument('--dataset', help='choice of dataset to generate jobs from. Currently, only "tpch" is supported', default='tpch')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='llama')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--rank', type=int, help='rank of low rank matrices training. if set to -1, low rank will not be enabled', default=128)
    # state encoder settings
    parser.add_argument('--pt-encoder-config', help='config file of pretrained state encoder. if not specified, create a new state encoder')
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder')
    # offline policy model related settings
    parser.add_argument('--K', type=int, help='horizon of offline policy to observe the past states, actions and returns (check the paper if you dont understand)', default=20)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--max-exec-num', type=int, default=50)
    parser.add_argument('--max-stage-num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-iters', type=int, default=10)
    parser.add_argument('--num-steps-per-iter', type=int, default=10000)
    parser.add_argument('--eval-max-ep-len', type=int, help='the max episode length of evaluation', default=6000)
    parser.add_argument('--eval-per-iter', type=int, help='evaluation per iteration', default=1)
    parser.add_argument('--save-checkpoint-per-iter', type=int, help='saving checkpoint per iteration')
    parser.add_argument('--target-return-scale', type=float, help='target return, see the original paper for details', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping: specify which layer to stop (layer index starts from 0)', default=-1)
    # other settings
    parser.add_argument('--train', action="store_true", help='train model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--env-seed', dest='env_seed', help='Environment random seed', type=int, default=None)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--resume-dir', help='model weight dir to resume training')
    parser.add_argument('--resume-iter', help='from which iteration to resume training', type=int)
    parser.add_argument('--model-dir', help='model weight dir for testing')
    parser.add_argument('--use-head', help='Use which head to predict next job stage', type=int, default=3)
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    
    args = parser.parse_args()

    # command examples:
    # python run_plm.py --train --test --grad-accum-steps 32 --seed 666 --plm-type llama --plm-size base --peft-rank 128 --device cuda:5 --device-out cuda:4 --state-feature-dim 256 --K 30 --gamma 1. --max-exec-num 50 --max-stage-num 100 --lr 0.0001 --warmup-steps 2000 --num-iters 40 --num-steps-per-iter 10000 --eval-max-ep-len 6000 --eval-per-iter 5 --save-checkpoint-per-iter 2 --target-return-scale 1
    # >>> if you don't want to use default experience pool, add arguments '--exp-pool-path your_exp_pool_path' <<<
    # >>> if you want to test on another environment settings, add arguments '--num-executors A --job-arrival-cap B' <<<
    # >>> if you want to test on the defautl environment settings, but on another environment, simply change the random seed will do '--seed your_seed' <<<
    # >>> if you want to use pretrained state encoder, add arguments '--pt-encoder-config your_pretrained_encoder_config_path --freeze-encoder (if you want to freeze state encoder)' <<<
    # >>> if you want to enable early stopping, add arguments '--which-layer your_stopping_layer (can be negative)', you may refer to PLM_LAYER_SIZES for the sizes of each plm's hidden layers <<<


    if args.device_out is None:  
        args.device_out = args.device

    if args.env_seed is None:
        args.env_seed = args.seed  # by default, env_seed is equal to random seed
    
    if args.save_checkpoint_per_iter is None:
        args.save_checkpoint_per_iter = args.eval_per_iter
    assert args.save_checkpoint_per_iter <= args.num_iters

    print('Arguments:')
    pprint(args)

    run(args)
