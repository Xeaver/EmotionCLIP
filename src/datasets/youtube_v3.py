"""
Movie dataset with human box annotations and sentiment logits.
"""

import os.path as osp
from typing import Optional, TypedDict
from collections.abc import Callable
import json
import math
from PIL import Image
import io
import logging

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm import tqdm
from rich import print as rprint
# import cv2
import orjson

from ..models.tokenizer import tokenize

logger = logging.getLogger()

# element in the dataset index
class SampleInfo(TypedDict):
    video_id: str
    duration: float
    start_frame: int
    end_frame: int
    text: str
    neutral_score: float
    has_person: float
    sentiment_logits: np.ndarray


# element returned by __getitem__    
class SampleData(TypedDict):
    video: torch.Tensor
    video_mask: torch.Tensor
    text: torch.Tensor
    sentiment_logits: torch.Tensor


# NOTE this is a dataset in 8 fps, numbers in brackets are the equivalent in 24 fps
# num_frames statistics. max: 79 (237), min: 15 (45), mean: 34 (102), std: 16 (48)
class MovieDataset(Dataset):
    def __init__(
        self, 
        video_path: str = '/ocean/projects/iri180005p/shared/video-data/video-sub/youtube', 
        caption_path: str = '/ocean/projects/iri180005p/shared/tv_cleaned',
        index_path: str = '/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_index.json',
        sentiment_path: str = '/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_sentiment_logits.npy',
        video_len: int = 8,
        target: str = 'token',
        tokenizer: Optional[Callable[[str], torch.Tensor]] = tokenize,
        frame_sampling_strategy: str = 'random',
        fps: int = 8,
        duration_min: int = 2,
        duration_max: int = 10,
        text_min: int = 3,
        neutral_score_threshold: float = 1.0,
        has_person_threshold: float = 0,
        preload_human_boxes: bool = True,
    ):
        assert osp.isdir(video_path), f'{video_path} is not a directory.'
        assert osp.isdir(caption_path), f'{caption_path} is not a directory.'
        assert osp.isfile(index_path), f'{index_path} is not a file.'
        assert osp.isfile(sentiment_path), f'{sentiment_path} is not a file.'
        assert target in ['text', 'token'], f'Invalid argument {target=}'
        if target == 'token':
            assert tokenizer is not None, f'Tokenizer is required when target is token.'
        assert frame_sampling_strategy in ['start', 'end', 'random'], f'Invalid argument {frame_sampling_strategy=}'
        assert duration_min * fps >= video_len, f'{duration_min=} is too short for {video_len=}.'
        assert 0 <= neutral_score_threshold <= 1, f'{neutral_score_threshold=} is out of range.'
        assert 0 <= has_person_threshold <= 1, f'{has_person_threshold=} is out of range.'
        
        super().__init__()
        # constants
        self.BBOX_TO_MASK_THRESHOLD = 0.5
        
        self.video_path = video_path
        self.caption_path = caption_path
        self.index_path = index_path
        self.sentiment_path = sentiment_path
        self.video_len = video_len
        self.target = target
        self.tokenizer = tokenizer
        self.frame_sampling_strategy = frame_sampling_strategy
        self.fps = fps
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.text_min = text_min
        self.neutral_score_threshold = neutral_score_threshold
        self.has_person_threshold = has_person_threshold
        self.preload_human_boxes = preload_human_boxes

        # created by self._create_index()
        self.video_ids: list[str]
        self.index: list[SampleInfo]
        self.human_boxes: dict[str, dict[str, list[list[float]]]]
        
        self._create_index()
        
        
    def _create_index(self):
        self.video_ids = set()
        self.index = []
        self.human_boxes = {}
        
        # load index file
        with open(self.index_path, 'r') as f:
            all_index: list[dict] = orjson.loads(f.read())
            
        # load sentiment logits
        all_sentiment_logits = np.load(self.sentiment_path)
        
        # create index for eligible video clips
        for i, sample_info in enumerate(tqdm(all_index, desc='Creating index')):
            criteria = (
                self.duration_min < sample_info['duration'] < self.duration_max,
                len(sample_info['text']) >= self.text_min,
                sample_info['neutral_score'] <= self.neutral_score_threshold,
                sample_info['has_person'] >= self.has_person_threshold,
            )
            if all(criteria):
                sample_info['sentiment_logits'] = all_sentiment_logits[i]
                self.index.append(sample_info)
                self.video_ids.add(sample_info['video_id'])
        self.video_ids = list(self.video_ids)
        logger.info(f'{len(self.index)} / {len(all_index)} samples after filtering.')
        
        # preload all human box annotations to speed up the __getitem__ function
        if self.preload_human_boxes:
            for video_id in tqdm(self.video_ids, desc='Loading human box annotations'):
                human_box_path = osp.join(self.video_path, video_id, 'human_boxes.json')
                with open(human_box_path, mode='rb') as f:
                    _human_box_data: list[dict[str, list[list[float]]]] = orjson.loads(f.read())
                human_box_data: dict[str, list[list[float]]] = {k: v for sample_dict in _human_box_data for k, v in sample_dict.items()}
                self.human_boxes[video_id] = human_box_data
        

    def __len__(self) -> int:
        return len(self.index)

    # for debugging purpose
    def __getitem__(self, i: int) -> SampleData:
        try:
            return self._get_item(i)
        except Exception as e:
            # logger.error(f'Failed to get item {i=}. {e}')
            print(f'Failed to get item {i=}. {e}', force=True)
            new_i = np.random.randint(len(self.index))
            return self.__getitem__(new_i)
        return self._get_item(i)
        
    # actual implementation of __getitem__
    def _get_item(self, i: int) -> SampleData:
        sample_info = self.index[i]

        # sparse sampling
        all_frame_indexes = list(range(sample_info['start_frame'], sample_info['end_frame']+1))
        if self.frame_sampling_strategy == 'random':
            sampled_frame_indexes = sorted(np.random.choice(all_frame_indexes, size=self.video_len, replace=False))
        else:
            return NotImplementedError

        # load video frames and get human boxes
        raw_frames = []
        raw_boxes = []
        with h5py.File(osp.join(self.video_path, sample_info['video_id'], 'frames.hdf5'), mode='r', swmr=True) as f:
            for frame_idx in sampled_frame_indexes:
                frame_name = self._idx_to_name(frame_idx)
                # load frame
                bimg = np.array(f[frame_name])
                img = Image.open(io.BytesIO(bimg))
                raw_frames.append(img)
                # load human box
                if self.preload_human_boxes:
                    human_box = self.human_boxes[sample_info['video_id']][frame_name]
                    raw_boxes.append(human_box)
                else:
                    # slow way to get human box
                    _human_box_path = osp.join(self.video_path, sample_info['video_id'], 'human_boxes.json')
                    with open(_human_box_path, mode='rb') as fh:
                        _human_box_data: list[dict[str, list[list[float]]]] = json.load(fh)
                    _human_box_data: dict[str, list[list[float]]] = {k: v for sample_dict in _human_box_data for k, v in sample_dict.items()}
                    human_box = _human_box_data[frame_name] # list[list[float]] or empty list
                    raw_boxes.append(human_box)
                
        video, video_mask = self._preprocess_frames(raw_frames, raw_boxes)    
        
        # preprocess text
        raw_text = sample_info['text']
        if self.target == 'text':
            text = raw_text
        elif self.target == 'token':
            text = self.tokenizer(raw_text).squeeze()
            
        sentiment_logits = torch.tensor(sample_info['sentiment_logits']).float()
            
        sample_data = SampleData(
            video=video,
            video_mask=video_mask,
            text=text,
            sentiment_logits=sentiment_logits,
        )
        return sample_data


    def _idx_to_name(self, n: int, width: int = 8) -> str:
        file_name = str(n).zfill(width) + '.jpg'
        return file_name
    
    
    # Scale the bboxes accordingly, since the original bboxes are extracted from the scaled images 
    def _get_correct_bbox(
        self, 
        boxes: list[list[float]], # [w1, h1, w2, h2, score]
        img_shape: tuple[int, int], # (w, h)
        yolo_shape: tuple[int, int] = (640, 640) # (w, h)
    ) -> list[list[float]]:
        # compute the scale ratio
        r = min(yolo_shape[0] / img_shape[0], yolo_shape[1] / img_shape[1])
        # map the human box onto the frame
        new_boxes = [
            [
                max(math.floor(box[0]/r), 0),
                max(math.floor(box[1]/r), 0),
                min(math.ceil(box[2]/r), img_shape[0]),
                min(math.ceil(box[3]/r), img_shape[1]),
                box[4]
            ]
            for box in boxes
        ]
        return new_boxes
    
    # list of boxes to single mask
    # boxes with score less than threshold are ignored
    def _bbox_to_mask(
        self, 
        boxes: list[list[float]], # [w1, h1, w2, h2, score], may be empty list
        target_shape: tuple[int, int], # (w, h),
        binary: bool = True, # whether to return binary mask or float mask
        binary_threshold: Optional[float] = 0.5
    ) -> torch.Tensor:
        mask = torch.zeros(target_shape[1], target_shape[0])
        if len(boxes) == 0:
            return mask
        
        if binary:
            for box in boxes:
                if box[4] > binary_threshold:
                   mask[box[1]:box[3], box[0]:box[2]] = 1
        else:
            mask = mask.unsqueeze(0).expand(len(boxes), *mask.shape)
            for i, box in enumerate(boxes):
                mask[i, box[1]:box[3], box[0]:box[2]] = box[4]
            mask = mask.max(dim=0)[0]
        return mask
    
    
    def _preprocess_frames(
        self, 
        frames: list[Image.Image],
        all_boxes: list[list[list[float]]],
        consistent: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # resize, box to mask
        frames = [F.resize(f, size=256, interpolation=transforms.InterpolationMode.BICUBIC) for f in frames]
        all_boxes = [self._get_correct_bbox(b, f.size) for f, b in zip(frames, all_boxes)]
        masks = [self._bbox_to_mask(b, f.size, binary_threshold=self.BBOX_TO_MASK_THRESHOLD) for f, b in zip(frames, all_boxes)]

        # random crop
        if consistent:
            i, j, h, w = transforms.RandomCrop.get_params(frame, output_size=(224, 224))
            processed_frames = [F.crop(frame, i, j, h, w) for frame in frames]
            processed_masks = [F.crop(mask, i, j, h, w) for mask in masks]
        else:
            processed_frames = []
            processed_masks = []
            for frame, mask in zip(frames, masks):
                i, j, h, w = transforms.RandomCrop.get_params(frame, output_size=(224, 224))
                frame = F.crop(frame, i, j, h, w)
                mask = F.crop(mask, i, j, h, w)
                processed_frames.append(frame)
                processed_masks.append(mask)
        
        # to tensor, normalize
        processed_frames = [
            F.normalize(
                tensor=F.to_tensor(frame), 
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ) 
            for frame in processed_frames
        ]
        
        video = torch.stack(processed_frames, dim=0).float()
        video_mask = torch.stack(processed_masks, dim=0).long()
        return video, video_mask


if __name__ == '__main__':
    pass
    