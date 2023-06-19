from __future__ import annotations
import os
import os.path as osp
from re import L
from typing import Literal, Optional
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
import orjson

from .utils import get_correct_bbox, bbox_to_mask
from ..models.tokenizer import tokenize

EMOTION_CLASS_NAMES = [
    'neutral',
    'surprise',
    'fear',
    'sadness',
    'joy',
    'disgust',
    'anger'
]

SENTIMENT_CLASS_NAMES = [
    'neutral',
    'positive',
    'negative'
]

logger = logging.getLogger()


# BUG train index 1165, no dir 'train_splits/dia125_utt3'
# BUG dev index 1084, no dir 'dev_splits/dia110_utt7'
# NOTE a lot of clip only have several frames
# num_frames statistics. max: 984, min: 2, mean: 75, std: 58
# n_sample with all / 32 / 64 frames: 
#   [train] 9988 / 8059 / 4454
#   [dev]   1108 /  892 / 496
#   [test]  2610 / 2160 / 1227
class MELD(Dataset):
    def __init__(
        self,
        data_dir: str = '/ocean/projects/iri180005p/psuzhang/data/MELD',
        split: Literal['train', 'dev', 'test'] = 'train',
        sampling_strategy: str = 'uniform',
        dense_sampling_interval: Optional[int] = 4,
        video_len: int = 8,
        target: Literal['utt_text', 'utt_token', 'emotion_idx', 'sentiment_idx', 'multimodal'] = 'emotion_idx',
    ):
        assert split in ['train', 'dev', 'test']
        
        super().__init__()
        # constants
        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        self.BBOX_TO_MASK_THRESHOLD = 0.5
        
        self.data_dir = data_dir
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.dense_sampling_interval = dense_sampling_interval
        self.video_len = video_len
        self.target = target
        
        self.index: pd.DataFrame
        self.all_human_boxes: dict
        self._create_index()
        
        # clip-style preprocesser
        self.preprocesser = T.Compose([
            T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        
    def _create_index(self):
        # load .csv data
        if self.split in ['train', 'dev', 'test']:
            annotation_file_path = osp.join(self.data_dir, 'MELD.Raw' ,f'{self.split}_sent_emo.csv')
            self.index = pd.read_csv(annotation_file_path)
        else:
            raise NotImplementedError

        # some preprocessing
        self.index['Emotion'] = self.index['Emotion'].apply(lambda x: EMOTION_CLASS_NAMES.index(x))
        self.index['Sentiment'] = self.index['Sentiment'].apply(lambda x: SENTIMENT_CLASS_NAMES.index(x))
        
        # mannually remove corrupted samples
        if self.split == 'train':
            CORRUPTED_SAMPLES_INFO = [
                (1165, 125, 3)
            ]
        elif self.split == 'dev':
            CORRUPTED_SAMPLES_INFO = [
                (1084, 110, 7)
            ]
        elif self.split == 'test':
            CORRUPTED_SAMPLES_INFO = []
        for idx, dia_id, utt_id in CORRUPTED_SAMPLES_INFO:
            assert self.index.loc[idx, 'Dialogue_ID'] == dia_id
            assert self.index.loc[idx, 'Utterance_ID'] == utt_id
            self.index.drop(idx, inplace=True)
        self.index.reset_index(drop=True, inplace=True)
        
        # load the human boxes
        boxes_fpath = osp.join(self.data_dir, f'{self.split}_human_boxes.json')
        with open(boxes_fpath, 'r') as f:
            self.human_boxes = orjson.loads(f.read())
        
        logger.info(f'Index of {self.split} set created, {self.index.shape[0]} samples in total.')
        
        
    def __len__(self):
        return self.index.shape[0]
    
    
    def __getitem__(self, i):
        dialogue_id = self.index.loc[i, 'Dialogue_ID']
        utterance_id = self.index.loc[i, 'Utterance_ID']
        clip_id = f'dia{dialogue_id}_utt{utterance_id}'
        clip_dir = osp.join(self.data_dir, 'frames', f'{self.split}_splits', clip_id)
        
        num_frames = len(os.listdir(clip_dir))
        # HACK return another sample if the clip is too short when using non-uniform sampling
        if self.sampling_strategy != 'uniform' and num_frames < self.video_len:
            new_i = np.random.randint(self.index.shape[0])
            return self.__getitem__(new_i)
        
        # sampling
        start_frame, end_frame = 0, num_frames-1
        if self.sampling_strategy == 'uniform':
            sampled_frame_ids = np.linspace(start_frame, end_frame, self.video_len, dtype=int)
        elif self.sampling_strategy == 'random':
            sampled_frame_ids = sorted(np.random.choice(np.arange(start_frame, end_frame+1), self.video_len, replace=False))

        # load video frames, human bboxes and process them
        frames = []
        masks = []
        for frame_id in sampled_frame_ids:
            frame_path = osp.join(clip_dir, f'frame_{frame_id}.jpg')
            raw_frame = Image.open(frame_path).convert('RGB')
            raw_boxes = self.human_boxes[clip_id][f'frame_{frame_id}.jpg']
            resized_frame = F.resize(raw_frame, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
            resized_boxes = get_correct_bbox(raw_boxes, resized_frame.size)
            mask = bbox_to_mask(resized_boxes, resized_frame.size, binary_threshold=self.BBOX_TO_MASK_THRESHOLD)
            frame = F.center_crop(resized_frame, self.CROP_SIZE)
            mask = F.center_crop(mask, self.CROP_SIZE)
            frame = F.normalize(
                tensor=F.to_tensor(frame),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            frames.append(frame)
            masks.append(mask)
        frames = torch.stack(frames, dim=0).float()
        video_masks = torch.stack(masks, dim=0).float()
        
        # load text
        if self.target == 'utt_text':
            target = self.index.loc[i, 'Utterance']
        elif self.target == 'utt_token':
            raw_text = self.index.loc[i, 'Utterance']
            target = tokenize(raw_text).squeeze()
        elif self.target == 'emotion_idx':
            target = self.index.loc[i, 'Emotion']
        elif self.target == 'sentiment_idx':
            target = self.index.loc[i, 'Sentiment']
        elif self.target == 'multimodal':
            target = {
                'utt_text': self.index.loc[i, 'Utterance'],
                'utt_token': tokenize(self.index.loc[i, 'Utterance']).squeeze(),
                'emotion_idx': self.index.loc[i, 'Emotion'],
            }
        else:
            raise NotImplementedError
        
        return frames, video_masks, target
        
        
if __name__ == '__main__':
    pass