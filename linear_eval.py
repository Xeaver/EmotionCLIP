import argparse
import os 
import os.path as osp
from dataclasses import dataclass
import logging
from typing import Literal, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, r2_score
from scipy.signal import savgol_filter
from rich import print as rprint

from src.models.base import EmotionCLIP
from src.datasets.meld import MELD
from src.datasets.bold import BoLD
from src.datasets.movie_graphs import MovieGraphsDataset
from src.datasets.emotic import Emotic
from src.datasets.liris_accede import LirisAccede
from src.engine.utils import set_random_seed
from src.engine.logger import setup_logger


@dataclass
class EvalArgs(argparse.Namespace):
    dataset: Literal['bold', 'mg', 'meld', 'emotic', 'la'] = 'bold'
    ckpt_path: str = './exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt'
    use_cache: bool = False # when use_cache=True, ckpt_path and save_path are ignored
    save_cache: bool = True
    
    ckpt_strict: bool = True
    cuda_deterministic: bool = True
    has_test_set: bool = False
    seed: int = 2022
    video_len: int = 8
    device: str = 'cuda:0'
    dataloader_workers: int = 4
    batch_size: int = 128
    cache_dir: str = './data/cache'


@torch.no_grad()
def extract_features(
    model: EmotionCLIP, 
    dataloader: DataLoader, 
    args: EvalArgs,
    split: str,
    data_type: str = 'video',
):
    feature_extractor = model.encode_video if data_type == 'video' else model.encode_image
    tdataloader = tqdm(dataloader, desc=f'Extracting features ({split})', unit_scale=dataloader.batch_size)
    all_features = []
    all_targets = []
    for i, batch in enumerate(tdataloader):
        # load batch
        visual_inputs, visual_masks, targets = batch
        visual_inputs = visual_inputs.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        visual_masks = visual_masks.to(args.device, non_blocking=True)
        # forward
        with torch.cuda.amp.autocast():
            features = feature_extractor(visual_inputs, visual_masks)
            features = F.normalize(features, dim=-1)
        all_features.append(features)
        all_targets.append(targets)
    all_features = torch.cat(all_features).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    return all_features, all_targets


def DefaultDataLoader(dataset: Dataset, args: EvalArgs):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_workers,
        pin_memory=True
    )

@torch.no_grad()
def accuracy_from_affect2mm(target: np.ndarray, output: np.ndarray):
    output = torch.from_numpy(output)
    target = torch.from_numpy(target)
    _, pred = output.topk(1, 1, True, True)
    target_value = torch.gather(target, 1, pred)
    correct_k = (target_value > 0).float().sum(0, keepdim=False).sum(0, keepdim=True)
    correct_k /= target.shape[0]
    res = correct_k.mul_(100.0).item()
    return res


def eval_liris_accede(model:EmotionCLIP, args: EvalArgs):
    train_dataset = LirisAccede(split='train')
    test_dataset = LirisAccede(split='test')
    train_dataloader = DefaultDataLoader(train_dataset, args)
    test_dataloader = DefaultDataLoader(test_dataset, args)
    X_train, y_train = extract_features(model, train_dataloader, args, split='train', data_type='video')
    X_test, y_test = extract_features(model, test_dataloader, args, split='test', data_type='video')

    reg = Ridge()
    results = {'mse': [], 'pcc': []}
    for i, dimension in enumerate(['valence', 'arousal']):
        reg.fit(X_train, y_train[:, i])
        y_pred = reg.predict(X_test)
        y_pred = savgol_filter(y_pred, window_length=100, polyorder=3)
        mse = mean_squared_error(y_test[:, i], y_pred)
        pcc = np.corrcoef(y_test[:, i], y_pred)[0, 1]
        results['mse'].append(mse)
        results['pcc'].append(pcc)
        print(f'{dimension}: mse={mse:.4f}, pcc={pcc:.4f}')
    print(f'Average: mse={np.mean(results["mse"]):.4f}, pcc={np.mean(results["pcc"]):.4f}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bold', choices=['bold', 'mg', 'meld', 'emotic', 'la'])
    parser.add_argument('--ckpt-path', type=str, default='./exps/cvpr_final-20221113-235224-a4a18adc/checkpoints/latest.pt')
    cargs = parser.parse_args()
    args = EvalArgs(
        dataset=cargs.dataset,
        ckpt_path=cargs.ckpt_path,
    )
    
    # basic setup
    set_random_seed(args.seed, deterministic=args.cuda_deterministic)
    logger = setup_logger(name='eval')
    rprint(args)
    
    if args.dataset == 'emotic':
        args.data_type = 'image'
    else:
        args.data_type = 'video'
    
    if not args.use_cache:
        # load pretrained model
        model = EmotionCLIP(
            video_len=args.video_len,
            backbone_checkpoint=None,
        )
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'], strict=args.ckpt_strict)
            model.eval().to(args.device)
            logger.info(f'Model loaded from {args.ckpt_path}')
        else:
            raise ValueError('No checkpoint provided')
        
        # create datasets and dataloaders
        if args.dataset == 'bold':
            train_dataset = BoLD(
                video_len=args.video_len,
                split='train'
            )
            val_dataset = BoLD(
                video_len=args.video_len,
                split='val'
            )
            test_dataset = None
        elif args.dataset == 'mg':
            train_dataset = MovieGraphsDataset(
                video_len=args.video_len,
                split='train'
            )
            val_dataset = MovieGraphsDataset(
                video_len=args.video_len,
                split='val'
            )
            test_dataset = MovieGraphsDataset(
                video_len=args.video_len,
                split='test'
            )
        elif args.dataset == 'meld':
            train_dataset = MELD(
                video_len=args.video_len,
                split='train',
                target='emotion_idx'
            )
            val_dataset = MELD(
                video_len=args.video_len,
                split='dev',
                target='emotion_idx'
            )
            test_dataset = MELD(
                video_len=args.video_len,
                split='test',
                target='emotion_idx'
            )
        elif args.dataset == 'emotic':
            train_dataset = Emotic(
                split='train',
            )
            val_dataset = Emotic(
                split='val',
            )
            test_dataset = Emotic(
                split='test',
            )
        elif args.dataset == 'la':
            eval_liris_accede(model, args)
            return
        else:
            raise ValueError(f'Unknown dataset {args.dataset}')
        args.has_test_set = test_dataset is not None
        train_dataloader = DefaultDataLoader(train_dataset, args)
        val_dataloader = DefaultDataLoader(val_dataset, args)
        test_dataloader = DefaultDataLoader(test_dataset, args) if args.has_test_set else None
        
        # extract features
        X_train, y_train = extract_features(model, train_dataloader, args, 'train', args.data_type)
        X_val, y_val = extract_features(model, val_dataloader, args, 'val', args.data_type)
        if args.has_test_set:
            X_test, y_test = extract_features(model, test_dataloader, args, 'test', args.data_type)
        logger.info('Features extracted')
            
        # cache features
        if args.save_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            with open(osp.join(args.cache_dir, f'{args.dataset}_train_features.npy'), 'wb') as f:
                np.save(f, X_train)
                np.save(f, y_train)
            with open(osp.join(args.cache_dir, f'{args.dataset}_val_features.npy'), 'wb') as f:
                np.save(f, X_val)
                np.save(f, y_val)
            if args.has_test_set:
                with open(osp.join(args.cache_dir, f'{args.dataset}_test_features.npy'), 'wb') as f:
                    np.save(f, X_test)
                    np.save(f, y_test)
            logger.info('Features cached')
    else:
        # load cached features
        with open(osp.join(args.cache_dir, f'{args.dataset}_train_features.npy'), 'rb') as f:
            X_train = np.load(f)
            y_train = np.load(f)
        with open(osp.join(args.cache_dir, f'{args.dataset}_val_features.npy'), 'rb') as f:
            X_val = np.load(f)
            y_val = np.load(f)
        test_cache_path = osp.join(args.cache_dir, f'{args.dataset}_test_features.npy')
        if osp.exists(test_cache_path):
            args.has_test_set = True
            with open(test_cache_path, 'rb') as f:
                X_test = np.load(f)
                y_test = np.load(f)
        logger.info('Cached features loaded')
        
    # setup linear classifier
    if args.dataset == 'bold':
        linear_clf = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=3.16,
            solver='sag',
            class_weight=None
        )
        linear_clf = OneVsRestClassifier(linear_clf)
    elif args.dataset == 'mg':
        linear_clf = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=10,
            solver='sag',
            class_weight=None
        )
        linear_clf = OneVsRestClassifier(linear_clf)
    elif args.dataset == 'meld':
        linear_clf = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=8,
            solver='sag',
            class_weight=None
        )
    elif args.dataset == 'emotic':
        linear_clf = LogisticRegression(
            random_state=args.seed,
            max_iter=2000,
            C=2.5,
            solver='sag',
            class_weight=None
        )
        linear_clf = OneVsRestClassifier(linear_clf)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    
    # train linear classifier
    linear_clf.fit(X_train, y_train)
    p_val = linear_clf.predict_proba(X_val)
    if args.has_test_set:
        p_test = linear_clf.predict_proba(X_test)
    
    if args.dataset == 'bold':
        # val
        mAP = average_precision_score(y_val, p_val, average='macro') * 100
        auc = roc_auc_score(y_val, p_val, average='macro') * 100
        logger.info(f'[BoLD val] mAP: {mAP:.2f} AUC: {auc:.2f}')
    elif args.dataset == 'mg':
        # val
        acc = accuracy_from_affect2mm(y_val, p_val)
        logger.info(f'[MovieGraphs val] acc: {acc:.2f}')
        # test
        acc = accuracy_from_affect2mm(y_test, p_test)
        logger.info(f'[MovieGraphs test] acc: {acc:.2f}')
    elif args.dataset == 'meld':
        p_val = np.argmax(p_val, axis=1)
        p_test = np.argmax(p_test, axis=1)
        # dev
        weighted_f1 = f1_score(y_val, p_val, average='weighted') * 100
        acc = accuracy_score(y_val, p_val) * 100
        logger.info(f'[MELD dev] weighted F1: {weighted_f1:.2f} acc: {acc:.2f}')
        # test
        weighted_f1 = f1_score(y_test, p_test, average='weighted') * 100
        acc = accuracy_score(y_test, p_test) * 100
        logger.info(f'[MELD test] weighted F1: {weighted_f1:.2f} acc: {acc:.2f}')
    elif args.dataset == 'emotic':
        # val
        mAP = average_precision_score(y_val, p_val, average='macro') * 100
        auc = roc_auc_score(y_val, p_val, average='macro') * 100
        logger.info(f'[Emotic val] mAP: {mAP:.2f} AUC: {auc:.2f}')
        # test
        mAP = average_precision_score(y_test, p_test, average='macro') * 100
        auc = roc_auc_score(y_test, p_test, average='macro') * 100
        logger.info(f'[Emotic test] mAP: {mAP:.2f} AUC: {auc:.2f}')
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')
    
    # predict VAD
    if args.dataset == 'bold':
        y_train, y_val = [], []
        for i in range(len(train_dataset)):
            target = train_dataset.annotations.loc[i, ['valence', 'arousal', 'dominance']].values.astype(np.float32)
            target = torch.from_numpy(target)
            y_train.append(target)
        for i in range(len(val_dataset)):
            target = val_dataset.annotations.loc[i, ['valence', 'arousal', 'dominance']].values.astype(np.float32)
            target = torch.from_numpy(target)
            y_val.append(target)
        y_train, y_val = np.stack(y_train), np.stack(y_val)
        
        reg = Ridge(alpha=1)
        reg.fit(X_train, y_train)
        p_val = reg.predict(X_val)
        r2 = r2_score(y_val, p_val, multioutput='uniform_average') * 100
        logger.info(f'[BoLD val] r2: {r2:.2f}')


if __name__ == '__main__':
    main()