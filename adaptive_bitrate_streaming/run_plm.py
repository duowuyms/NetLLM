import os
import sys
import numpy as np
import torch
import pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pprint import pprint
from munch import Munch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import cfg
from baseline_special.utils.utils import load_traces
from baseline_special.utils.constants import BITRATE_LEVELS
from plm_special.trainer import Trainer
from plm_special.evaluate import evaluate_on_env
from plm_special.test import test_on_env
from plm_special.data.dataset import ExperienceDataset
from plm_special.models.rl_policy import OfflineRLPolicy
from plm_special.models.state_encoder import EncoderNetwork
from plm_special.models.low_rank import peft_model
from plm_special.utils.utils import set_random_seed
from plm_special.utils.plm_utils import load_plm
from plm_special.utils.console_logger import ConsoleLogger


PLM_LAYER_SIZES = {
    'gpt2': {
        'base': 24,
        'small': 12,
        'large': 36,
        'xl': 48
    },
    'llama': {
        'base': 32,
    },
    't5-lm': { 
        'base': 12,
        'small': 6,
        'large': 24,
        'xl': 24
    }
}


def save_model(args, model, save_dir):
    if args.rank > 0:
        # save lora weights
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(), os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # lora is disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    if args.rank > 0:
        # load lora weights
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # lora is disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def adapt(args, model, exp_dataset, exp_dataset_info, eval_env_settings, checkpoint_dir, best_model_dir, eval_process_reward_fn):
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.warmup_steps, 1)
    )
    loss_fn = CrossEntropyLoss()
    trainer = Trainer(args, model=model, optimizer=optimizer, exp_dataset=exp_dataset, loss_fn=loss_fn, device=args.device, lr_scheduler=lr_scheduler, 
                      grad_accum_steps=args.grad_accum_steps)

    target_return = exp_dataset_info.max_return * args.target_return_scale
    best_eval_return = 0.

    total_train_losses = []
    for epoch in range(args.num_epochs):
        train_logs, train_losses = trainer.train_epoch()
        total_train_losses.extend(train_losses)
        print('='* 20, f'Training Iteration #{epoch}', '=' * 20)
        print('>' * 10, 'Training Information:')
        pprint(train_logs)

        if epoch % args.save_checkpoint_per_epoch == 0:  # save checkpoint
            checkpoint_dir_epoch = os.path.join(checkpoint_dir, str(epoch))
            if not os.path.exists(checkpoint_dir_epoch):
                os.makedirs(checkpoint_dir_epoch)
            save_model(args, model, checkpoint_dir_epoch)
            print('Checkpoint saved at:', checkpoint_dir_epoch)

        if epoch % args.eval_per_epoch == 0:
            eval_logs = evaluate_on_env(args, env_settings=eval_env_settings, model=model, target_return=target_return, max_ep_num=args.trace_num,
                                        process_reward_fn=eval_process_reward_fn)
            episodes_return = eval_logs['episodes_return']
            if best_eval_return < episodes_return:
                best_eval_return = episodes_return
                save_model(args, model, best_model_dir)
                print('Best model saved at:', best_model_dir)

            eval_logs['best_return'] = best_eval_return
            print('>' * 10, 'Evaluation Information')
            pprint(eval_logs)
    # save training losses
    train_losses_path = os.path.join(checkpoint_dir, 'train_losses.txt')
    np.savetxt(train_losses_path, total_train_losses, fmt='%.6f', delimiter='\n')


def test(args, model, exp_dataset_info, env_settings, model_dir, result_dir, test_process_reward_fn):
    model = load_model(args, model, model_dir)
    print('Load model from:', model_dir)
    target_return = exp_dataset_info.max_return * args.target_return_scale
    results = test_on_env(args, model, result_dir, env_settings, target_return, args.trace_num, test_process_reward_fn, seed=args.seed)
    print(results)
    print('Test time:', results['time'], '\nMean reward:', results['mean_reward'])
    print('Results saved at:', result_dir)


def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes
    assert args.exp_pool_path is not None, 'please specify a experience pool path for training'
    assert args.trace in cfg.trace_dirs.keys()
    assert args.video in cfg.video_size_dirs.keys()

    # 1. set seed
    set_random_seed(args.seed)

    # 2. create environment setting
    trace_dir = cfg.trace_dirs[args.trace]
    video_size_dir = cfg.video_size_dirs[args.video]
    all_cooked_time ,all_cooked_bw ,all_file_names, all_mahimahi_ptrs = load_traces(trace_dir)
    args.trace_num = min(args.trace_num, len(all_file_names))
    if args.trace_num == -1:
        args.trace_num = len(all_file_names)
    if args.trace_num == len(all_file_names):
        args.fixed_order = True

    env_settings = {
        'all_cooked_time': all_cooked_time,
        'all_cooked_bw': all_cooked_bw,
        'all_file_names': all_file_names,
        'all_mahimahi_ptrs': all_mahimahi_ptrs,
        'video_size_dir': video_size_dir,
        'fixed': args.fixed_order,
        'trace_num': args.trace_num,
    }

    # 3. create training dataset, fetch info
    exp_pool = pickle.load(open(args.exp_pool_path, 'rb'))
    exp_dataset = ExperienceDataset(exp_pool, gamma=args.gamma, scale=args.scale, max_length=args.w, sample_step=args.sample_step)
    exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
    print('Experience dataset info:')
    pprint(exp_dataset_info)
    
    # 4. create model
    
    # 4.1 load plm
    # args.device_out and args.device_mid are used for model parallelism (currently only support llama) 
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None). 
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    plm, *_ = load_plm(args.plm_type, os.path.join(cfg.plm_dir, args.plm_type, args.plm_size), 
                       device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)

    if args.plm_type != 'llama':
        plm = plm.to(args.device)
    
    if args.rank != -1:
        plm = peft_model(plm, args.plm_type, rank=args.rank)

    # 4.2 create state encoder
    assert args.state_feature_dim is not None, 'please specify state feature dim to create state encoder'
    state_encoder = EncoderNetwork(embed_dim=args.state_feature_dim)
    state_encoder = state_encoder.to(args.device)

    # 4.3 create rl policy
    plm_embed_size = cfg.plm_embed_sizes[args.plm_type][args.plm_size]
    max_ep_len = exp_dataset_info.max_timestep + 1
    rl_policy = OfflineRLPolicy(state_feature_dim=args.state_feature_dim, bitrate_levels=BITRATE_LEVELS, state_encoder=state_encoder, plm=plm, plm_embed_size=plm_embed_size, 
                                           max_length=args.w, max_ep_len=max_ep_len, device=args.device, device_out=args.device_out, which_layer=args.which_layer)

    # 5. handling directory and path

    # extract training experience pool information
    train_exp_pool_info = args.exp_pool_path.split('/')[-4:-1]
    train_exp_pool_info = '_'.join(train_exp_pool_info)
    models_dir = os.path.join(cfg.plm_ft_dir, f'{args.plm_type}_{args.plm_size}', train_exp_pool_info + f'_ss_{args.sample_step}', f'rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_sfd_{args.state_feature_dim}'\
                              f'_lr_{args.lr}_wd_{args.weight_decay}_warm_{args.warmup_steps}_epochs_{args.num_epochs}_seed_{args.seed}')
    results_dir = os.path.join(cfg.results_dir, f'{args.trace}_{args.video}', f'trace_num_{args.trace_num}_fixed_{args.fixed_order}', f'{args.plm_type}_{args.plm_size}',
                               f'early_stop_{args.which_layer}_rank_{args.rank}_w_{args.w}_gamma_{args.gamma}_tgt_scale_{args.target_return_scale}_seed_{args.seed}')
    checkpoint_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_checkpoint')
    best_model_dir = os.path.join(models_dir, f'early_stop_{args.which_layer}_best_model')


    # 6. start training/testing
    def process_reward(reward, 
                       max_reward=exp_dataset_info.max_reward, 
                       min_reward=exp_dataset_info.min_reward, 
                       scale=args.scale):
        reward = min(max_reward, max(min_reward, reward))  # bound reward
        return (reward - min_reward) / (max_reward - min_reward) / scale
    
    torch.backends.cudnn.benchmark = True

    if args.adapt:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        console_log = open(os.path.join(models_dir, f'early_stop_{args.which_layer}_console.log'), 'w')
        sys.stdout = ConsoleLogger(sys.__stdout__, console_log)
        adapt(args, rl_policy, exp_dataset, exp_dataset_info, env_settings, checkpoint_dir, best_model_dir, process_reward)
    if args.test:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        model_dir = args.model_dir if args.model_dir is not None else best_model_dir
        assert os.path.exists(model_dir), f'Model weight dir {model_dir} does not exist.'
        test(args, rl_policy, exp_dataset_info, env_settings, model_dir, results_dir, process_reward)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # training dataset settings
    parser.add_argument('--exp-pool-path', help='the path storing the experience pool file for training', default='artifacts/exp_pools/exp_pool.pkl')
    parser.add_argument('--sample-step', type=int, help='the steps for sampling experiences')
    # environment settings
    parser.add_argument('--trace', help='name of traces (e.g., fcc-test)', type=str, default='fcc-test')
    parser.add_argument('--trace-num', help='number of traces. if set to -1, use all traces in the trace dir.', type=int, default=100)
    parser.add_argument('--video', help='name of video (e.g., video1)', type=str, default='video1')
    parser.add_argument('--fixed-order', action='store_true', help='iterate over test traces in a fixed sequential order.')
    # plm settings
    parser.add_argument('--plm-type', type=str, default='gpt2')
    parser.add_argument('--plm-size', type=str, default='base')
    parser.add_argument('--rank', type=int, help='rank of low-rank matrices. if set to -1, low-rank matrices will not be enabled', default=-1)
    # state encoder settings
    parser.add_argument('--state-feature-dim', type=int, help='feature dim of the state encoder', default=256)
    # rl policy related settings
    parser.add_argument('--w', type=int, help='context window for learning return distribution', default=20)
    parser.add_argument('--gamma', type=float, help='discounted factor of reward', default=1.)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--eval-per-epoch', type=int, help='evaluation per epoch', default=1)
    parser.add_argument('--save-checkpoint-per-epoch', type=int, help='saving checkpoint per iteration')
    parser.add_argument('--target-return-scale', type=float, help='target return, which specifies the expected performance for the model to achieve', default=1.)
    parser.add_argument('--which-layer', type=int, help='for early stopping (not used in our experiments): specify which layer to stop (layer index starts from 0)', default=-1)
    # other settings
    parser.add_argument('--adapt', action="store_true", help='adapt model')
    parser.add_argument('--test', action="store_true", help='test model')
    parser.add_argument('--grad-accum-steps', dest='grad_accum_steps', type=int, default=32)
    parser.add_argument('--seed', help='random seed', type=int, default=100003)
    parser.add_argument('--scale', help='scale reward/return', type=int, default=1000)
    parser.add_argument('--model-dir', help='model weight dir for testing')
    parser.add_argument('--device', action='store', dest='device', help='device (cuda or cpu) to run experiment')
    parser.add_argument('--device-out', action='store', dest='device_out', help='device (cuda or cpu) to place the split of model near the output')
    parser.add_argument('--device-mid', action='store', dest='device_mid', help='device (cuda or cpu) to place the split of model between the input and output')
    
    args = parser.parse_args()

    # >>> for debug <<<
    # args.exp_pool_path = 'artifacts/exp_pools/exp_pool.pkl'
    # args.plm_type = 'llama'
    # args.plm_size = 'base'
    # args.rank = 128
    # args.state_feature_dim = 256
    # args.num_epochs = 1
    # args.eval_per_epoch = 1
    # args.adapt = True
    # args.test = True
    # args.device = 'cuda:0'
    # args.device_out = 'cuda:0'
    # args.which_layer = -1
    # args.seed = 100003
    # >>> for debug <<<

    # command examples:
    # python run_plm.py --adapt --test --grad-accum-steps 32 --seed 666 --plm-type llama --plm-size base --rank 128 --device cuda:0 --state-feature-dim 256 --w 20 --gamma 1. --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 --target-return-scale 1
    # >>> if you want to use your own experience pool, add arguments '--exp-pool-path your_exp_pool_path' <<<
    # >>> if you want to use your own trace dataset, add arguments '--trace your_trace --trace-num number_of_traces --fixed-order (if you want to iterate over all traces in a fixed sequential order)' <<<
    # >>> if you want to use your own video dataset, add arguments '--video your_video'<<<
    # >>> if you want to enable early stopping, add arguments '--which-layer your_stopping_layer (can be negative)', you may refer to PLM_LAYER_SIZES for the sizes of each plm's hidden layers <<<


    if args.device_out is None:  
        args.device_out = args.device
    
    if args.save_checkpoint_per_epoch is None:
        args.save_checkpoint_per_epoch = args.eval_per_epoch
    assert args.save_checkpoint_per_epoch <= args.num_epochs

    print('Arguments:')
    pprint(args)

    run(args)
