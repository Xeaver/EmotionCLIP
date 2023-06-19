import os
import os.path as osp
from typing import Callable, Optional, Literal
import math
from operator import itemgetter
import math

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from rich import print as rprint
# import cv2

from ..models.tokenizer import tokenize


CLASS_NAMES = [
    'peace',
    'affection',
    'esteem',
    'anticipation',
    'engagement',
    'confidence',
    'happiness',
    'pleasure',
    'excitement',
    'surprise',
    'sympathy',
    'doubt confusion',
    'disconnection',
    'fatigue',
    'embarrassment',
    'yearning',
    'disapproval',
    'aversion',
    'annoyance',
    'anger',
    'sensitivity',
    'sadness',
    'disquietment',
    'fear',
    'pain',
    'suffering'
]


# 1D array to [w1, h1, w2, h2]
def joints_to_bbox(
    arr: np.ndarray, 
    x_padding: int = 0, 
    y_padding: int = 0
)-> tuple[int, list[float]]:
    f_id, _, joints = arr[0], arr[1], arr[2:].reshape(18, 3)
    joints = joints[~np.all(joints == 0, axis=1)]
    x_min, y_min, _ = joints.min(axis=0)
    x_max, y_max, _ = joints.max(axis=0)
    bbox = (
        int(f_id), 
        [
            math.floor(y_min) - y_padding, 
            math.floor(x_min) - x_padding, 
            math.ceil(y_max) + y_padding, 
            math.ceil(x_max) + x_padding
        ]
    )
    return bbox

# [w1, h1, w2, h2] to binary mask
def bbox_to_mask(bbox: list[float], target_shape: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(target_shape[1], target_shape[0])
    if len(bbox) == 0:
        return mask
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
    return mask


# num_frames statistics. max: 299, min: 101, mean: 167, std: 53
class BoLD(Dataset):
    def __init__(self, 
        data_folder: str = '/ocean/projects/iri180005p/shared/BOLD_public',
        video_len: int = 8,
        split: Literal['train', 'val'] = 'train',
        sampling_strategy: str = 'uniform_all',
        target: str = 'idx',
        template: Optional[Callable] = None
    ):
        super().__init__()
        # constants
        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        
        self.data_folder = data_folder
        self.video_len = video_len
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.target = target
        self.template = template if template else lambda x: x
        self.class_names = CLASS_NAMES
        
        self.video_folder = osp.join(self.data_folder, 'mmextract')
        self.annotation_folder = osp.join(self.data_folder, 'annotations')
        self.joint_folder = osp.join(self.data_folder, 'joints')
        
        # load annotations
        if self.split in ['train', 'val']:
            self.annotations = pd.read_csv(
                osp.join(self.annotation_folder, f'{self.split}.csv'),
                header=None,
                names= (
                    ['video_id', 'person_id', 'start_frame', 'end_frame'] +
                    self.class_names + 
                    ['valence', 'arousal', 'dominance', 'gender', 'age', 'ethinicity', 'confidence_score']
                )
            )
        else:
            raise NotImplementedError
        self.annotations[self.class_names] = (self.annotations[self.class_names] >= 0.5).astype(int)
        
        self.all_bboxes: list[dict[int, list[float]]]
        self._generate_all_bboxes()
        
    def _load_all_joints(self) -> dict[str, list[np.ndarray]]:
        all_joints = {}
        for video_id in tqdm(self.annotations['video_id'].unique(), desc='Loading joints'):
            joints = np.load(osp.join(self.joint_folder, video_id[:-4]+'.npy'))
            joints[:, 0] = joints[:, 0] - min(joints[:, 0]) + 1
            all_joints[video_id] = joints
        return all_joints
    
    def _generate_all_bboxes(self):
        all_joints = self._load_all_joints()
        self.all_bboxes = []
        for i in tqdm(range(len(self.annotations)), desc='Generating bounding boxes'):
            video_id = self.annotations.loc[i, 'video_id']
            joints = all_joints[video_id]
            person_joints = joints[joints[:, 1] == self.annotations.loc[i, 'person_id']]
            bboxes = dict(np.apply_along_axis(joints_to_bbox, 1, person_joints))
            self.all_bboxes.append(bboxes)
            
        
    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, i):
        video_id = self.annotations.loc[i, 'video_id']
        start_frame = self.annotations.loc[i, 'start_frame']
        end_frame = self.annotations.loc[i, 'end_frame']
        clip_folder = osp.join(self.video_folder, video_id)
        
        all_frame_names = os.listdir(clip_folder)
        if self.sampling_strategy == 'random_all':
            frames_to_be_used = sorted(np.random.choice(all_frame_names, size=self.video_len, replace=False))
        elif self.sampling_strategy == 'uniform_all':
            frames_to_be_used = itemgetter(*np.linspace(0, len(all_frame_names)-1, self.video_len, dtype=int).tolist())(sorted(all_frame_names))
        else:
            raise NotImplementedError
        frames = []
        bboxes = []
        for frame_name in frames_to_be_used:
            frame = Image.open(osp.join(clip_folder, frame_name))
            bbox = self.all_bboxes[i].get(self._fname_to_id(frame_name), [])
            frames.append(frame)
            bboxes.append(bbox)
        video, video_mask = self._preprocess_frames(frames, bboxes)
        
        # get labels (sparse)
        if self.target == 'idx':
            target = self.annotations.loc[i, self.class_names].values.astype(np.float32)
            target = torch.from_numpy(target)
        elif self.target == 'vad':
            target = self.annotations.loc[i, ['valence', 'arousal', 'dominance']].values.astype(np.float32)
            target = torch.from_numpy(target)
        return video, video_mask, target
    
    
    def _preprocess_frames(
        self,
        frames: list[Image.Image],
        bboxes: list[list[float]],
        crop_method: str = 'center',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        RESIZE_SIZE, CROP_SIZE = self.RESIZE_SIZE, self.CROP_SIZE
        processed_frames = []
        processed_masks = []
        for frame, bbox in zip(frames, bboxes):
            mask = bbox_to_mask(bbox, frame.size)
            resized_frame = F.resize(frame, size=RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)
            resized_mask = F.resize(mask.unsqueeze(0), size=RESIZE_SIZE).squeeze()
            if crop_method == 'center':
                cropped_frame = F.center_crop(resized_frame, output_size=CROP_SIZE)
                cropped_mask = F.center_crop(resized_mask, output_size=CROP_SIZE)
            elif crop_method == 'random':
                i, j, h, w = transforms.RandomCrop.get_params(resized_frame, output_size=CROP_SIZE)
                cropped_frame = F.crop(resized_frame, i, j, h, w)
                cropped_mask = F.crop(resized_mask, i, j, h, w)
            else:
                raise NotImplementedError
            normalized_frame = F.normalize(
                tensor=F.to_tensor(cropped_frame), 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            binarized_mask = (cropped_mask > 0.5).long()
            processed_frames.append(normalized_frame)
            processed_masks.append(binarized_mask)
            
        video = torch.stack(processed_frames, dim=0).float()
        video_mask = torch.stack(processed_masks, dim=0).long()
        return video, video_mask
            
    
    def _fname_to_id(self, fname: str) -> int:
        return int(fname.split('.')[0].split('_')[1])

