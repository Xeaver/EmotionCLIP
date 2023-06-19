import argparse
import json
import logging
import math
import os
import os.path as osp
import sys
import time
from contextlib import suppress
from typing import Optional, Any
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..options import DefaultArgs
from ..models.loss import ClipLoss
from ..models.loss_v2 import ReweightedClipLoss
from .distributed import is_master
from .utils import unwrap_model, AverageMeter
from .lr_scheduler import LRWarmupScheduler
from .evaluation import EvaluatorBase


class TrainerBase:
    def __init__(
        self, 
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel, 
        dataloader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        grad_scaler: Optional[torch.cuda.amp.GradScaler], 
        lr_scheduler: Optional[LRWarmupScheduler], 
        args: DefaultArgs,
        evaluator: Optional[EvaluatorBase] = None,
        eval_freq: Optional[int] = None # times of eval per epoch
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.evaluator = evaluator
        self.eval_freq = eval_freq
        
        # precompute evaluation steps
        if evaluator:
            self.eval_steps_per_epoch = np.linspace(0, len(dataloader)-1, self.eval_freq+1).astype(int)[1:] if eval_freq else [len(dataloader)-1]
        else:
            self.eval_steps_per_epoch = []

        # init logger and monitor
        self.logger = logging.getLogger()
        self.monitor = {
            'loss': AverageMeter(),
            'batch_time': AverageMeter(),
            'data_time': AverageMeter()
        }

        # training status
        self.current_epoch: int
        self.current_step: int


    def train_one_epoch(self, epoch: int) -> None:
        self.logger.info(f'Epoch {epoch} started.')
        
        self.current_epoch = epoch
        autocast = torch.cuda.amp.autocast if self.args.precision == 'amp' else suppress

        # setup model
        self.model.train()

        # setup loss
        if self.args.loss_reweight_scale is None:
            criterion = ClipLoss(
                local_loss=self.args.local_loss,
                gather_with_grad=self.args.gather_with_grad,
                cache_labels=True,
                rank=self.args.rank,
                world_size=self.args.world_size
            )
        else:
            criterion = ReweightedClipLoss(
                local_loss=self.args.local_loss,
                gather_with_grad=self.args.gather_with_grad,
                cache_labels=True,
                rank=self.args.rank,
                world_size=self.args.world_size,
                sentiment_scale=self.args.loss_reweight_scale
            )

        # setup dataloader
        if self.args.distributed and self.dataloader.sampler is not None:
            self.dataloader.sampler.set_epoch(self.current_epoch)

        # training loop
        end_time = time.time()
        with tqdm(self.dataloader, desc=f'Epoch {self.current_epoch}', unit_scale=self.dataloader.batch_size) as pbar:
            for i, batch in enumerate(pbar):
                self.current_step = len(self.dataloader) * self.current_epoch + i

                if len(batch) == 2:
                    videos, texts = batch['video'], batch['text']
                elif len(batch) == 3:
                    videos, video_masks, texts  = batch['video'], batch['video_mask'], batch['text']
                elif len(batch) == 4:
                    videos, video_masks, texts, sentiment_logits = batch['video'], batch['video_mask'], batch['text'], batch['sentiment_logits']
                videos = videos.to(device=self.args.device, non_blocking=True)
                video_masks = video_masks.to(device=self.args.device, non_blocking=True)
                texts = texts.to(device=self.args.device, non_blocking=True)
                sentiment_logits = sentiment_logits.to(device=self.args.device, non_blocking=True)

                self.logger.debug(f'Batch has been transfered to {self.args.device}')

                self.monitor['data_time'].update(time.time() - end_time)

                self.model.train()
                self.optimizer.zero_grad()

                # forward pass
                with autocast():
                    video_features, text_features, logit_scale = self.model(videos, video_masks, texts)
                    if self.args.loss_reweight_scale is None:
                        loss = criterion(video_features, text_features, logit_scale)
                    else:
                        loss = criterion(video_features, text_features, logit_scale, sentiment_logits)

                self.logger.debug('Forward pass finished.')

                # backward pass
                if self.grad_scaler:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.iter_step()

                self.logger.debug('Backward pass finished.')

                # clamp to 4.6052 = ln(100), as in the original paper.
                with torch.no_grad():
                    unwrap_model(self.model).logit_scale.clamp_(0, math.log(100))

                self.monitor['loss'].update(loss.item())
                self.monitor['batch_time'].update(time.time() - end_time)
                end_time = time.time()

                # show smoothed loss on progress bar
                pbar.set_postfix({'loss': self.monitor['loss'].avg(5)})

                # wandb log loss
                if is_master(self.args) and self.args.wandb and self.current_step % self.args.wandb_log_steps == 0:
                    loss_dict = {
                        'training_step': self.current_step,
                        'epoch': self.current_epoch,
                        'loss': self.monitor['loss'].avg(5)
                    }
                    wandb.log(loss_dict)
                self.logger.debug('Wandb log finished.')
                
                # evaluation
                if is_master(self.args) and i in self.eval_steps_per_epoch:
                    metrics = self.evaluator.eval()
                    self.logger.debug(f'Evaluation finished. {self.current_step=}')
                    metrics['training_step'] = self.current_step
                    metrics['epoch'] = self.current_epoch
                    metrics['loss'] = self.monitor['loss'].avg(5)
                    self.logger.info(metrics)
                    if self.args.wandb:
                        wandb.log(metrics)

                # only train 5 iter in debug mode
                if self.args.debug_mode and i >= 5:
                    break
            # -- end for --
        # -- end tqdm wrapper --

        # update lr_scheduler
        if self.lr_scheduler:
            self.lr_scheduler.epoch_step()

        # wandb log batch_loss
        if is_master(self.args) and self.args.wandb:
            batch_loss_dict = {
                'batch_loss': self.monitor['loss'].avg(),
                'epoch': self.current_epoch
            }
            wandb.log(batch_loss_dict)

        self.logger.info(f"batch loss: {self.monitor['loss'].avg():.2f}")
        self.logger.info(f"data time:  {self.monitor['data_time'].avg():.2f}")
        self.logger.info(f"batch time: {self.monitor['batch_time'].avg():.2f}")

        # reset time monitor per epoch
        self.monitor['loss'].reset()
        self.monitor['data_time'].reset()
        self.monitor['batch_time'].reset()

        self.logger.debug(f'Training epoch {self.current_epoch} finished.')

            


    def save_checkpoint(self, file_name: str) -> None:
        if not is_master(self.args):
            # self.logger.warning(f'Saving checkpoint on non-master process {self.rank=}')
            return

        # construct checkpoint
        data = {}
        data['args'] = vars(self.args)
        data['epoch'] = self.current_epoch
        data['model'] = unwrap_model(self.model).state_dict()
        data['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            data['grad_scaler'] = self.grad_scaler.state_dict()
        if self.lr_scheduler:
            data['lr_scheduler'] = self.lr_scheduler.state_dict()

        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        file_path = osp.join(self.args.ckpt_dir, file_name)
        torch.save(data, file_path)

        # tag the latest checkpoint
        latest = osp.join(self.args.ckpt_dir, 'latest.pt')
        torch.save(data, latest)

        self.logger.info(f"Save checkpoint to {file_path}.")


    # TODO
    def load_checkpoint(self, ckpt_path: str, model_only: bool = False, override: bool = False, map_location: str = 'cpu') -> None:
        assert os.path.isfile(ckpt_path), f'Not found {ckpt_path=}'
        self.logger.info(f"Loading checkpoint from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location=map_location)

        if model_only:
            # check args consistency
            ckpt_args = argparse.Namespace(**checkpoint['args'])
            self.check_args_consistency(ckpt_args, override=override)

            # load model
            incompatible = unwrap_model(self.model).load_state_dict(checkpoint['model'], strict=False)
            if incompatible.missing_keys:
                self.logger.warning("Encounter missing keys when loading model weights:\n"
                                    f"{incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                self.logger.warning("Encounter unexpected keys when loading model weights:\n"
                                    f"{incompatible.unexpected_keys}")
        else:
            # check args consistency
            ckpt_args = argparse.Namespace(**checkpoint['args'])
            self.check_args_consistency(ckpt_args, override=override)

            # load model
            incompatible = unwrap_model(self.model).load_state_dict(checkpoint['model'], strict=False)
            if incompatible.missing_keys:
                self.logger.warning("Encounter missing keys when loading model weights:\n"
                                    f"{incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                self.logger.warning("Encounter unexpected keys when loading model weights:\n"
                                    f"{incompatible.unexpected_keys}")

            # load epoch
            self.args.start_epoch = checkpoint['epoch'] + 1

            # load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # load grad_scaler
            if 'grad_scaler' in checkpoint.keys():
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler'])

            # load lr_scheduler
            if 'lr_sceduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info(f"Load checkpoint from {ckpt_path}.")


    def check_args_consistency(self, ckpt_args: argparse.Namespace, override: bool = False) -> None:
        # skip device id
        ignore = ['rank', 'local_rank', 'device']
        curr_dict, ckpt_dict = self.args.__dict__, ckpt_args.__dict__
        for k in set().union(curr_dict.keys(), ckpt_dict.keys()):
            if k in ignore or (k in curr_dict.keys() and k in ckpt_dict.keys() and curr_dict[k] == ckpt_dict[k]):
                continue
            elif k in curr_dict.keys() and k in ckpt_dict.keys() and curr_dict[k] != ckpt_dict[k]:
                self.logger.warning(f'Incosistent argument: args.{k}={curr_dict[k]}, ckpt.{k}={ckpt_dict[k]}')
                if override:
                    setattr(self.args, k, ckpt_dict[k])
            elif k in curr_dict.keys() and k not in ckpt_dict.keys():
                self.logger.warning(f'Incosistent argument: args.{k}={curr_dict[k]}, ckpt.{k} not exists')
                if override:
                    delattr(self.args, k)
            elif k not in curr_dict.keys() and k in ckpt_dict.keys():
                self.logger.warning(f'Incosistent argument: args.{k} not exists, ckpt.{k}={ckpt_dict[k]}')
                if override:
                    setattr(self.args, k, ckpt_dict[k])
        if override:
            self.logger.info('Override `args` from the checkpoint.')
