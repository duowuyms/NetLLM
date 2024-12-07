import numpy as np
import torch
import time

from munch import Munch

from plm_special.utils import (process_batch_actions, process_batch_returns, process_batch_timesteps)


class Trainer:
    def __init__(self, args, model, optimizer, exp_dataset, loss_fn, device, use_head, batch_size=1, grad_accum_steps=1, lr_scheduler=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.exp_dataset = exp_dataset
        self.loss_fn = loss_fn
        self.device = device
        self.use_head = use_head
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.lr_scheduler = lr_scheduler
        
        # self.dataloader = DataLoader(exp_dataset, batch_size, shuffle=True, pin_memory=True)
        self.exp_dataset_info = Munch(exp_dataset.exp_dataset_info)
        self.diagnostics = dict()
        self.start_time = time.time()

    def train_iteration(self, num_steps, report_loss_per_steps=100):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for step in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss.item())

            # perform gradient accumulation update
            train_loss = train_loss / self.grad_accum_steps
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            if ((step + 1) % self.grad_accum_steps == 0) or (step + 1 == num_steps):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            if step % report_loss_per_steps == 0:                
                mean_train_loss = np.mean(train_losses)
                print(f'Step {step} - mean train loss {mean_train_loss:>9f}')

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        return logs, train_losses

    def train_step(self):
        states, actions, returns, timesteps = self.exp_dataset.sample_batch(self.batch_size)
        actions, targe_stage_actions, targ_exec_actions = process_batch_actions(actions, max_stage_num=self.args.max_stage_num, max_exec_num=self.args.max_exec_num, device=self.device)
        returns = process_batch_returns(returns, device=self.device)
        timesteps = process_batch_timesteps(timesteps, device=self.device)

        stage_preds1, stage_preds2, exec_preds = self.model(states, actions, returns, timesteps, stage_indices=targe_stage_actions, use_head=self.use_head)

        exec_preds = exec_preds.permute(0, 2, 1)
        loss = self.loss_fn(exec_preds, targ_exec_actions[:, :exec_preds.shape[2]])

        if stage_preds1 is not None:
            loss1 = []
            for i in range(len(stage_preds1)):
                stage_pred1 = stage_preds1[i]
                loss1.append(self.loss_fn(stage_pred1, targe_stage_actions[0, i][:, :stage_pred1.shape[2]]))
            loss1 = torch.mean(torch.tensor(loss1))
            loss = loss + loss1
        if stage_preds2 is not None:
            stage_preds2 = stage_preds2.permute(0, 2, 1)
            loss2 = self.loss_fn(stage_preds2, targe_stage_actions[:, :stage_preds2.shape[2]])
            loss = loss + loss2

        return loss
