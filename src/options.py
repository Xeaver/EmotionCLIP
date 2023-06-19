import argparse
from dataclasses import dataclass
from optparse import Option
from typing import Optional, Any, List, Tuple
import os
import os.path as osp
import logging
import sys
from datetime import datetime
import uuid

import torch
import numpy as np

from .engine.distributed import world_info_from_env

logger = logging.getLogger()

@dataclass
class DefaultArgs(argparse.Namespace):
    # exp info
    exp_id: str # should only be used in master process
    debug_mode: bool | str = False
    dummy_train_data: bool = False
    dummy_eval_data: bool = False

    # general setting
    PROJECT_PATH: str = '.'
    device_mode: str = 'cuda'
    device: Optional[str] = None
    precision: str = 'amp'
    seed: Optional[int] = 2022
    deterministic: bool = False

    # distributed
    distributed: Optional[bool] = None
    dist_backend: str = 'nccl'
    dist_url: str = 'env://'
    no_set_device_rank: bool = False # should be True only when CUDA_VISIBLE_DEVICES restricted to one per proc
    use_bn_sync: bool = True
    ddp_static_graph: bool = True
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None

    # dataset
    video_len: int = 8
    sampling_strategy: str = 'random'
    neutral_score_threshold: float = 0.05
    sampling_strategy_bold: str = 'uniform_all'
    preload_human_boxes : bool = True
    

    # dataloader
    batch_size: int = 128
    train_loader_workers: int = 8
    val_loader_workers: int = 4
    pin_memory: bool = True

    # model
    backbone_config: str = osp.join(PROJECT_PATH, 'src/models/model_configs/ViT-B-32.json')
    backbone_checkpoint: str = osp.join(PROJECT_PATH, 'src/pretrained/vit_b_32-laion2b_e16-af8dbd0c.pth')
    temporal_fusion: str = 'transformer'
    head_nlayer: int = 6

    # loss
    local_loss: bool = False
    gather_with_grad: bool = False
    loss_reweight_scale: Optional[float] = 10

    # training
    start_epoch: int = 0
    max_epochs: int = 25
    lr_backbone_gb: float = 5e-5
    lr_backbone_rest: float = 1e-8
    lr_head_gb: float = 5e-5
    lr_head_rest: float = 1e-8
    lr_min: float = 1e-10
    weight_decay_backbone: float = 0.1
    weight_decay_head: float = 0.1
    adamw_beta1: float = 0.98
    adamw_beta2: float = 0.9
    adamw_eps: float = 1e-6
    warmup_by_epoch: bool = False
    warmup_t: int = 10000
    warmup_mode: str = 'auto'
    warmup_init_lr: Optional[float] = None
    warmup_init_factor: Optional[float] = 1e-5
    reset_logit_scale: bool = False

    # evaluation
    enable_eval: bool = True
    eval_freq: int = 1

    # misc
    # NOTE should only be used in master process
    log_level: int = logging.INFO
    ckpt_dir: bool | str = True # save ckpt or not, pass str to override default path
    log_dir: bool | str = True # same
    save_script: bool = True

    # wandb
    wandb: bool = True
    wandb_watch: bool = False
    wandb_log_steps: int = 100

    # slurm job id
    slurm_job_id: Optional[str] = None

    # flag used to filter results on wandb
    version: str = 'cvpr2023'


    def __post_init__(self):
        # get process id
        self.local_rank, self.rank, self.world_size = world_info_from_env()

        # debug mode setting
        if self.debug_mode is True:
            self.exp_id = 'debug'
            self.log_level = logging.NOTSET
            self.wandb = False
            self.max_epochs = 5
            self.train_loader_workers = 2
            self.val_loader_workers = 2
            self.dummy_train_data = True
            self.dummy_eval_data = True
        elif self.debug_mode == 'eval':
            self.exp_id = 'debug'
            self.wandb = False
            self.max_epochs = 0
            self.dummy_train_data = True
            self.dummy_eval_data = False

        # create a unique exp_id and folder for this experiment
        if self.rank == 0:
            uid = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + str(uuid.uuid4()).split('-')[0]
            self.exp_id = self.exp_id + '-' + uid
            exp_folder = osp.join(self.PROJECT_PATH, 'exps', self.exp_id)
            os.makedirs(exp_folder, exist_ok=False)

            # setup default path
            if self.ckpt_dir is True:
                self.ckpt_dir = osp.join(exp_folder, 'checkpoints')
            if self.log_dir is True:
                self.log_dir = osp.join(exp_folder, 'logs')

        # generate seed if neccesory
        if self.seed is None:
            self.seed = np.random.randint(1, 2**31)

        # save slurm_job_id if possible
        if 'SLURM_JOB_ID' in os.environ:
            self.slurm_job_id = os.environ['SLURM_JOB_ID']

        # sanity check
        assert self.device_mode in ('cuda', 'cpu'), f'Invalid {self.device_mode=}'
        assert self.precision in ('amp', 'fp32'), f'Invalid {self.precision=}'

