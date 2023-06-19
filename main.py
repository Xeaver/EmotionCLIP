import argparse
import logging
import os.path as osp
import sys
import shutil
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import yaml
import numpy as np
from tqdm import tqdm
from rich import print as rprint

from src.models.base import EmotionCLIP
from src.datasets.youtube_v3 import MovieDataset
from src.datasets.bold import BoLD
from src.datasets.utils import DummyDataset
from src.engine.trainer import TrainerBase
from src.engine.evaluation import MulticlassLinearClassifier
from src.engine.lr_scheduler import LRWarmupScheduler
from src.engine.utils import set_random_seed, PlaceholderModule
from src.engine.distributed import is_master, init_distributed_device, world_info_from_env, setup_print_for_distributed
from src.engine.logger import setup_logger
from src.options import DefaultArgs


def main():
    ##################
    #     Config     #
    ##################
    args = DefaultArgs(
        exp_id='default',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path to the video dataset folder.')
    parser.add_argument('--caption-path', type=str, help='Path to the caption dataset folder.')
    parser.add_argument('--sentiment-path', type=str, help='Path to the precomputed sentiment logits file.')
    parser.add_argument('--index-path', type=str, help='Path to the index file.')
    cargs = parser.parse_args()
    for k, v in vars(cargs).items():
        setattr(args, k, v)

    ###################
    #   Basic setup   #
    ###################
    # 1. setup distributed training
    init_distributed_device(args)
    setup_print_for_distributed(args)

    # 2. setup logging
    logger = setup_logger(
        name=None,
        log_level=args.log_level,
        rank=args.rank,
        color=True,
        output_dir=args.log_dir
    )

    # 3. Make sure each worker has a different, yet deterministic seed
    if args.distributed:
        set_random_seed(seed=args.seed + args.rank, deterministic=args.deterministic)
    else:
        set_random_seed(seed=args.seed, deterministic=args.deterministic)

    # 4. backup code and config
    if args.save_script and is_master(args):
        # save script
        exp_folder = osp.join(args.PROJECT_PATH, 'exps', args.exp_id)
        shutil.copy(__file__, osp.join(exp_folder, 'main.py'))
        # save args
        with open(osp.join(exp_folder, 'args.yaml'), mode='w') as f:
            yaml.safe_dump(vars(args), f)
    if not args.distributed:
        rprint(vars(args))
    else:
        logger.debug(f'{vars(args)=}')


    ################
    #   Training   #
    ################
    # setup model and dataloader
    model = EmotionCLIP(
        backbone_config=args.backbone_config,
        backbone_checkpoint=args.backbone_checkpoint,
        temporal_fusion=args.temporal_fusion,
        video_len=args.video_len,
        head_nlayer=args.head_nlayer,
        reset_logit_scale=args.reset_logit_scale,
    ).to(args.device)
    logger.info('Model created.')
            
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            module=model,
            device_ids=[args.device] if args.device_mode=='cuda' else None,
            static_graph=args.ddp_static_graph
        )
        logger.info('DDP model created.')


    # setup dataloader
    if args.dummy_train_data:
        train_dataset = DummyDataset({
            'video': ((args.video_len, 3, 224, 224), torch.float),
            'video_mask': ((args.video_len, 224, 224), torch.float),
            'text': (77, torch.long),
            'sentiment_logits': (7, torch.float),
        })
    else:
        train_dataset = MovieDataset(
            video_path=args.video_path,
            caption_path=args.caption_path,
            sentiment_path=args.sentiment_path,
            index_path=args.index_path,
            video_len=args.video_len,
            frame_sampling_strategy=args.sampling_strategy,
            neutral_score_threshold=args.neutral_score_threshold,
            preload_human_boxes=args.preload_human_boxes,
        )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.train_loader_workers,
        pin_memory=args.pin_memory
    )
    logger.info('Train dataloader created.')
    

    # setup optimizer, grad_scaler and lr_scheduler
    # helper filters
    is_frame_params = lambda n, p: 'backbone.visual' in n
    is_text_params = lambda n, p: 'backbone' in n and 'visual' not in n and 'logit_scale' not in n
    is_temporal_params = lambda n, p: 'visual_head' in n and 'logit_scale' not in n
    is_gain_or_bias_params = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    # paramter groups
    all_named_parameters = list(model.named_parameters())
    frame_gb_params = [
        p for n, p in all_named_parameters if is_frame_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    frame_rest_params = [
        p for n, p in all_named_parameters if is_frame_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    text_gb_params = [
        p for n, p in all_named_parameters if is_text_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    text_rest_params = [
        p for n, p in all_named_parameters if is_text_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    temporal_gb_params = [
        p for n, p in all_named_parameters if is_temporal_params(n, p) and is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    temporal_rest_params = [
        p for n, p in all_named_parameters if is_temporal_params(n, p) and not is_gain_or_bias_params(n, p) and p.requires_grad
    ]
    logit_scale_params = [
        p for n, p in all_named_parameters if 'logit_scale' in n and p.requires_grad
    ]
    # setup optimizer
    param_groups_for_optimizer = [
        {'params': frame_gb_params, 'lr': args.lr_backbone_gb, 'weight_decay': 0.},
        {'params': frame_rest_params, 'lr': args.lr_backbone_rest, 'weight_decay': args.weight_decay_backbone},
        {'params': text_gb_params, 'lr': args.lr_backbone_gb, 'weight_decay': 0.},
        {'params': text_rest_params, 'lr': args.lr_backbone_rest, 'weight_decay': args.weight_decay_backbone},
        {'params': temporal_gb_params, 'lr': args.lr_head_gb, 'weight_decay': 0.},
        {'params': temporal_rest_params, 'lr': args.lr_head_rest, 'weight_decay': args.weight_decay_head},
        {'params': logit_scale_params, 'lr': args.lr_backbone_rest, 'weight_decay': 0.}
    ]
    optimizer = torch.optim.AdamW(
        param_groups_for_optimizer,
        betas = (args.adamw_beta1, args.adamw_beta2),
        eps = args.adamw_eps
    )
    # setup grad_scaler for AMP
    grad_scaler = torch.cuda.amp.GradScaler() if args.precision == 'amp' else None
    # setup lr_scheduler
    wrapped_lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.lr_min,
    )
    lr_scheduler = LRWarmupScheduler(
        wrapped_lr_scheduler,
        by_epoch=True,
        epoch_len=len(train_dataloader),
        warmup_t=args.warmup_t,
        warmup_by_epoch=args.warmup_by_epoch,
        warmup_mode=args.warmup_mode,
        warmup_init_lr=args.warmup_init_lr,
        warmup_init_factor=args.warmup_init_factor,
    )
    logger.info('Optimizer, lr_scheduler and grad_scaler created.')


    ################
    #  Evaluation  #
    ################
    if args.enable_eval and is_master(args):
        if not args.dummy_eval_data:
            finetune_dataset = BoLD(
                video_len=args.video_len,
                split='train',
                sampling_strategy=args.sampling_strategy_bold
            )

            val_dataset = BoLD(
                video_len=args.video_len,
                split='val',
                sampling_strategy=args.sampling_strategy_bold
            )
        else:
            finetune_dataset = DummyDataset(
                [
                    ((args.video_len, 3, 224, 224), torch.float), # video
                    ((args.video_len, 224, 224), torch.float), # video_mask
                    (26, torch.float), # label
                ],
                length=1000
            )
            val_dataset = deepcopy(finetune_dataset)
            
        finetune_dataloader = DataLoader(
            dataset=finetune_dataset,
            batch_size=args.batch_size,
            sampler=None,
            shuffle=False,
            drop_last=False,
            num_workers=args.val_loader_workers,
            pin_memory=args.pin_memory
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            sampler=None,
            shuffle=False,
            drop_last=False,
            num_workers=args.val_loader_workers,
            pin_memory=args.pin_memory
        )
        linear_clf = MulticlassLinearClassifier(
            model=model,
            dataloaders=[finetune_dataloader, val_dataloader],
            args=args
        )
        logger.info('Evaluation protocal created on master process.')


    ###############
    #  Main loop  #
    ###############
    # placeholders
    if not args.distributed:
        global dist
        dist = PlaceholderModule()
    if not args.wandb:
        global wandb
        wandb = PlaceholderModule()

    # create trainer
    trainer = TrainerBase(
        model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        lr_scheduler=lr_scheduler,
        args=args,
        evaluator=linear_clf if args.enable_eval and is_master(args) else None,
        eval_freq=args.eval_freq
    )

    # wandb init
    if is_master(args):
        wandb.init(
            name=args.exp_id,
            project="Movie-CLIP",
            config=vars(args)
        )
        if args.wandb_watch:
            wandb.watch(model, log='all')

    logger.info('Main loop started.')

    # main loop
    for current_epoch in range(args.start_epoch, args.max_epochs):
        trainer.train_one_epoch(current_epoch)
        if args.ckpt_dir:
            trainer.save_checkpoint(file_name=f'epoch{current_epoch}.pt')
        dist.barrier()

    logger.info('Job completed')
    if is_master(args):
        wandb.finish()
    sys.exit()


if __name__ == '__main__':
    main()