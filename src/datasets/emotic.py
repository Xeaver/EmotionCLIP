from __future__ import annotations
import os
import os.path as osp
from typing import Literal, Optional
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


def get_correct_bbox(
    box: list[float],
    img_shape: tuple[int, int],
    orginal_shape: tuple[int, int]
) -> list[float]:
    r = min(orginal_shape[0] / img_shape[0], orginal_shape[1] / img_shape[1])
    new_box = [
        max(math.floor(box[0]/r), 0),
        max(math.floor(box[1]/r), 0),
        min(math.ceil(box[2]/r), img_shape[0]),
        min(math.ceil(box[3]/r), img_shape[1]),
    ]
    return new_box


class Emotic(Dataset):
    def __init__(
        self,
        data_dir: str = '/ocean/projects/iri180005p/psuzhang/data/emotic',
        split: Literal['train', 'val', 'test'] = 'train',
        target: Literal['cat', 'cont'] = 'cat',
        preprocesser: Optional[nn.Module] = None,
        video_len: Optional[int] = None,
    ):
        super().__init__()
        self.RESIZE_SIZE = 224
        self.CROP_SIZE = 224
        
        self.data_dir = data_dir
        self.split = split
        self.target = target
        self.video_len = video_len
        
        # load annotations
        annotation_dir = osp.join(data_dir, 'annotations_pre')
        self.index = pd.read_csv(osp.join(annotation_dir, f'{split}.csv'))
        self.cat_labels: np.ndarray = np.load(osp.join(annotation_dir, f'{split}_cat_arr.npy'))
        self.cont_labels: np.ndarray = np.load(osp.join(annotation_dir, f'{split}_cont_arr.npy'))

        # preprocess
        self.index['Image Size'] = self.index['Image Size'].apply(eval)
        self.index['BBox'] = self.index['BBox'].apply(eval)
        
        
        # clip-style preprocesser
        if preprocesser:
            self.preprocesser = preprocesser
        else:
            self.preprocesser = T.Compose([
                T.Resize(size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=self.CROP_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, i):
        sample_info = self.index.loc[i]
        
        # process img and generate mask
        img_file_path = osp.join(self.data_dir, 'emotic', sample_info['Folder'], sample_info['Filename'])
        img = Image.open(img_file_path).convert('RGB')
        original_size = sample_info['Image Size']
        bbox = sample_info['BBox']
        img = F.resize(img, size=self.RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
        bbox = get_correct_bbox(bbox, img.size, original_size)
        mask = torch.zeros(img.size[1], img.size[0])
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        img = F.center_crop(img, self.CROP_SIZE)
        mask = F.center_crop(mask, self.CROP_SIZE)
        img = F.normalize(
            tensor=F.to_tensor(img),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).float()
        mask = mask.float()
        
        if self.video_len is not None:
            img = img.repeat(self.video_len, 1, 1, 1)
            mask = mask.repeat(self.video_len, 1, 1)
        
        if self.target == 'cat':
            target = self.cat_labels[i]
        elif self.target == 'cont':
            target = self.cont_labels[i]
        target = torch.from_numpy(target).float()
    
        return img, mask, target