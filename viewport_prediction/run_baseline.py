import re

import numpy as np
import torch
import os
import sys
import argparse
import random
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from config import cfg
from dataset.load_dataset import create_dataset
from utils.models_utils import create_model
from utils.normalize import normalize_data, denormalize_data
from utils.result_notebook import ResultNotebook


def track_train(args, model, dataloader_train, dataloader_valid, models_dir):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_'\
        f'epochs_{args.epochs}_bs_{args.bs}_lr_{args.lr}_seed_{args.seed}'
    
    train_size = len(dataloader_train)
    valid_size = len(dataloader_valid)

    best_model_path = os.path.join(models_dir, 'best_model_' + file_prefix + '.pth')
    folder_path = os.path.dirname(best_model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_valid_loss, best_epoch = float('inf'), 0
    print(f'Training {args.model} on {args.train_dataset} - bs: {args.bs} - lr: {args.lr} - seed: {args.seed}')

    total_loss = []

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}\n-------------------------------")
        model.train()
        
        for batch,(r_his, r_fut, r_his_images, r_fut_images, _) in enumerate(dataloader_train):
            gt = r_fut.to(args.device)
            r_his, r_fut = r_his.to(args.device), r_fut[:, 0:1, :].to(args.device)
            his, fut = normalize_data(r_his, args.train_dataset), \
               normalize_data(r_fut, args.train_dataset)
            his_images = torch.stack(r_his_images, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)
            fut_images = torch.stack(r_fut_images, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)

            pred = model(his, his_images, fut, fut_images)
            train_loss = criterion(pred, gt)

            total_loss.append(float(train_loss))
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            print(f"\rTrain: [{batch + 1}/{train_size}] - train_loss: {train_loss:>9f}", end='')
        print('\n',f'Train: mean train loss: {(sum(total_loss)/len(total_loss)):>9f}')
        
        # Validate
        if epoch % args.epochs_per_valid == 0:
            model.eval() 
            with torch.no_grad(): 
                total_valid_loss = []
                for batch, (r_his_val, r_fut_val, r_his_images_val, r_fut_images_val, _) in enumerate(dataloader_valid):
                    gt_val = r_fut_val.to(args.device)
                    r_his_val, r_fut_val = r_his_val.to(args.device), r_fut_val[:, 0:1, :].to(args.device)
                    his_val, fut_val = normalize_data(r_his_val, args.train_dataset), \
                        normalize_data(r_fut_val, args.train_dataset)
                    his_images_val = torch.stack(r_his_images_val, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)
                    fut_images_val = torch.stack(r_fut_images_val, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)

                    pred_val = model(his_val, his_images_val, fut_val, fut_images_val)
                    valid_loss = criterion(pred_val, gt_val)
                    total_valid_loss.append(valid_loss)

                mean_valid_loss = sum(total_valid_loss) / len(total_valid_loss)
                print(f'Valid: mean valid loss: {mean_valid_loss:>9f}')
                if best_valid_loss > mean_valid_loss:
                    best_valid_loss = mean_valid_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path) 
                    print(f'Best model (epoch {best_epoch}, loss {best_valid_loss}) saved at', best_model_path)


def test(args, model, dataloader_test, models_dir, results_dir):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_'\
                  f'epochs_{args.epochs}_bs_{args.bs}_lr_{args.lr}_seed_{args.seed}'
    best_model_path = os.path.join(models_dir, 'best_model_' + file_prefix + '.pth') if args.model_path is None else args.model_path
    result_path = os.path.join(results_dir, 'result_' + file_prefix + '.csv')
    notebook = ResultNotebook()

    if args.model not in ['regression', 'velocity']:  # linear regression/velocity doesn't need loading model weights
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        print('Load model from', best_model_path)

    print(f'Testing {args.model} on {args.test_dataset} - seed: {args.seed}')
    with torch.no_grad():
        for raw_history, raw_future, info in tqdm(dataloader_test):
            raw_history, raw_future = raw_history.to(args.device), raw_future.to(args.device)
            history, future = normalize_data(raw_history, args.test_dataset), \
                normalize_data(raw_future, args.test_dataset)
            pred, gt = model.inference(history, future)
            pred, gt = denormalize_data(pred, args.test_dataset), \
                denormalize_data(gt, args.test_dataset)
            videos, users, timesteps = info[0], info[1], info[2]
            notebook.record(pred, gt, videos, users, timesteps)
        notebook.write(result_path)


def track_test(args, model, dataloader_test,models_dir, results_dir): 
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_ss_{args.sample_step}_'\
                  f'epochs_{args.epochs}_bs_{args.bs}_lr_{args.lr}_seed_{args.seed}'
    best_model_path = os.path.join(models_dir, 'best_model_' + file_prefix + '.pth') if args.model_path is None else args.model_path
    result_path = os.path.join(results_dir, 'result_' + file_prefix + '.csv')

    notebook = ResultNotebook()
    
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        print('Load model from', best_model_path)
    
    print(f'Testing {args.model} on {args.test_dataset} - seed: {args.seed}')
    with torch.no_grad():
        for batch, (r_his_test, r_fut_test, r_his_images_test, r_fut_images_test, info) in enumerate(dataloader_test):
            gt_test = r_fut_test.to(args.device)
            r_his_test, r_fut_test = r_his_test.to(args.device), r_fut_test[:, 0:1, :].to(args.device)
            his_test, fut_test = normalize_data(r_his_test, args.train_dataset), \
                 normalize_data(r_fut_test, args.train_dataset)
            his_images_test = torch.stack(r_his_images_test, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)
            fut_images_test = torch.stack(r_fut_images_test, dim=0).permute(1, 0, 2, 3).unsqueeze(-1).to(args.device)
            pred_test = model(his_test, his_images_test, fut_test, fut_images_test)
            videos, users, timesteps = info[0], info[1], info[2]
            notebook.record(pred_test, gt_test, videos, users, timesteps)
        notebook.write(result_path)


def run(args):
    assert args.train_dataset in cfg.dataset_list 
    assert args.test_dataset in cfg.dataset_list
    assert args.model in ['regression', 'velocity', 'track']

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    models_dir = os.path.join(cfg.models_dir, args.model, args.train_dataset, f'{args.dataset_frequency}Hz')
    results_dir = os.path.join(cfg.results_dir, args.model, args.test_dataset, f'{args.dataset_frequency}Hz')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model = create_model(args.model, args.his_window, args.fut_window, args.device, args.seed).to(args.device)
    
    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        model = torch.compile(model).to(args.device)  # recommend to compile model when you are using PyTorch 2.0
    
    torch.set_float32_matmul_precision('high')

    if args.train:
        dataset_train, dataset_valid = create_dataset(args.train_dataset, his_window=args.his_window, fut_window=args.fut_window,
                                                    frequency=args.dataset_frequency, step=args.sample_step, trim_head=args.trim_head, 
                                                    trim_tail=args.trim_tail, include=['train', 'valid'], for_track=True)
        dataloader_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, pin_memory=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=args.bs, shuffle=False, pin_memory=True)
        track_train(args, model, dataloader_train, dataloader_valid, models_dir)

    if args.test:
        if args.model == 'track':
            dataset_test = create_dataset(args.test_dataset, his_window=args.his_window, fut_window=args.fut_window, step=args.sample_step,
                                      frequency=args.dataset_frequency, trim_head=args.trim_head, trim_tail=args.trim_tail, include=['test'], for_track=True)[0]
            dataloader_test = DataLoader(dataset_test, batch_size=args.bs, shuffle=False, pin_memory=True)
            track_test(args, model, dataloader_test,models_dir, results_dir)
        else:
            dataset_test = create_dataset(args.test_dataset, his_window=args.his_window, fut_window=args.fut_window, step=args.sample_step,
                                        frequency=args.dataset_frequency, trim_head=args.trim_head, trim_tail=args.trim_tail, include=['test'])[0]
            dataloader_test = DataLoader(dataset_test, batch_size=args.bs, shuffle=True, pin_memory=True)
            test(args, model, dataloader_test, models_dir, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

    # ========== model/plm settings related arguments ==========
    parser.add_argument('--train', action="store_true", help='Train model.')
    parser.add_argument('--test', action="store_true", help='Test model.')
    parser.add_argument('--device', action='store', dest='device', help='Device (cuda or cpu) to run experiment.')
    parser.add_argument('--model', action='store', dest='model', help='Model type, e.g., track.')
    parser.add_argument('--compile', action='store_true', dest='compile', 
                        help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        help='(Optional) Resume model weights from checkpoint for training.')
    
    # ========== dataset settings related arguments ==========
    parser.add_argument('--train-dataset', action='store', dest='train_dataset', help='Dataset for training.')
    parser.add_argument('--test-dataset', action='store', dest='test_dataset', help='Dataset for testing.')
    
    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--his-window', action='store', dest='his_window',
                        help='(Optional) Historical window', type=int)
    parser.add_argument('--fut-window', action='store', dest='fut_window',
                        help='(Optional) Future (prediction) window.', type=int)
    parser.add_argument('--trim-head', action='store', dest='trim_head',
                        help='(Optional) Trim some part of the viewport trajectory head.', type=int)
    parser.add_argument('--trim-tail', action='store', dest='trim_tail',
                        help='(Optional) Trim some part of the viewport trajectory tail.', type=int)
    parser.add_argument('--dataset-frequency', action='store', dest='dataset_frequency',
                        help='(Optional) The frequency version of the dataset.', type=int)
    parser.add_argument('--sample-step', action='store', dest='sample_step',
                        help='(Optional) The steps for sampling viewports.', type=int)
    
    # ========== training related settings ==========
    parser.add_argument('--epochs', action="store", dest='epochs', help='(Optional) Neural network learning epochs.', type=int)
    parser.add_argument('--epochs-per-valid', action='store', dest='epochs_per_valid', type=int,
                        help='(Optional) The number of epochs per validation (default 5).')
    parser.add_argument('--lr', action="store", dest='lr', help='(Optional) Neural network learning rate.', type=float)
    parser.add_argument('--weight-decay', action="store", dest='weight_decay', help='(Optional) Neural network weight decay.', type=float)
    parser.add_argument('--bs', action="store", dest='bs', help='(Optional) Batch size.', type=int)
    parser.add_argument('--model-path', action="store", dest='model_path', help='(Optional) Model checkpoint path.', type=str)
    parser.add_argument('--seed', action="store", dest='seed', type=int, default=1,
                        help='(Optional) Random seed (default to 1).')
    args = parser.parse_args()

    # for debug --- start
    # args.train = False
    # args.test = True
    # args.device = 'cuda:0'
    # args.train_dataset = 'Jin2022'
    # args.test_dataset = 'Jin2022'
    # args.model = 'track'
    # args.epochs = 1
    # args.bs = 32  #banch size
    # args.compile = True
    # args.model_path = '/data-NVMeSSD/wuduo/notmuch/projects/2023_prompt_learning/NetLLM/viewport_prediction/data/models/track/pretrain.pth'
    # args.lr = 0.0005
    # for debug --- end

    # handle defautl settings
    args.his_window = cfg.default_history_window if args.his_window is None else args.his_window
    args.fut_window = cfg.default_future_window if args.fut_window is None else args.fut_window
    args.trim_head = cfg.default_trim_head if args.trim_head is None else args.trim_head
    args.trim_tail = cfg.default_trim_tail if args.trim_tail is None else args.trim_tail
    args.epochs = cfg.default_epochs if args.epochs is None else args.epochs
    args.lr = cfg.default_lr if args.lr is None else args.lr
    args.bs = cfg.default_bs if args.bs is None else args.bs
    args.epochs_per_valid = cfg.default_epochs_per_valid if args.epochs_per_valid is None else args.epochs_per_valid
    args.dataset_frequency = cfg.default_dataset_frequency if args.dataset_frequency is None else args.dataset_frequency
    args.sample_step = cfg.default_sample_step if args.sample_step is None else args.sample_step

    if args.model in ['regression', 'velocity']:
        args.train = False
        args.compile = False
        args.device = 'cpu'
        print(f'Detect model: {args.model}. Automatically disenable train and compile mode and set device to cpu.')

    if args.train_dataset is None:
        args.train_dataset = args.test_dataset
    if args.test_dataset is None:
        args.test_dataset = args.train_dataset

    # command example:
    # python run_models.py --model track --train --test --device cuda:2 --train-dataset Jin2022 --test-dataset Jin2022 --lr 0.0005 --bs 64 --epochs 80 --seed 1 --compile --device cuda:2 --his-window 10 --fut-window 20 --dataset-frequency 5 --sample-step 15

    print(args)
    run(args)

