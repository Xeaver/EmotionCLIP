from __future__ import annotations
import os
import os.path as osp
from typing import Callable, Optional, Literal, TypedDict
import math
import logging

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import orjson

from .utils import get_correct_bbox, bbox_to_mask


EMOTION_LABELS = {
    'Affection': ['loving', 'friendly'], 
    'Anger': ['anger', 'furious', 'resentful', 'outraged', 'vengeful'],
    'Annoyance': ['annoy', 'frustrated', 'irritated', 'agitated', 'bitter', 'insensitive', 'exasperated', 'displeased'],
    'Anticipation':	['optimistic', 'hopeful', 'imaginative', 'eager'],
    'Aversion':	['disgusted', 'horrified', 'hateful'],
    'Confidence':	['confident', 'proud', 'stubborn', 'defiant', 'independent', 'convincing'],
    'Disapproval':	['disapproving', 'hostile', 'unfriendly', 'mean', 'disrespectful', 'mocking', 'condescending', 'cunning', 'manipulative', 'nasty', 'deceitful', 'conceited', 'sleazy', 'greedy', 'rebellious', 'petty'],
    'Disconnection':	['indifferent', 'bored', 'distracted', 'distant', 'uninterested', 'self-centered', 'lonely', 'cynical', 'restrained', 'unimpressed', 'dismissive'],
    'Disquietment':	['worried', 'nervous', 'tense', 'anxious','afraid', 'alarmed', 'suspicious', 'uncomfortable', 'hesitant', 'reluctant', 'insecure', 'stressed', 'unsatisfied', 'solemn', 'submissive'],
    'Doubt/Conf':	['confused', 'skeptical', 'indecisive'],
    'Embarrassment':	['embarrassed', 'ashamed', 'humiliated'],
    'Engagement':	['curious', 'serious', 'intrigued', 'persistent', 'interested', 'attentive', 'fascinated'],
    'Esteem':	['respectful', 'grateful'],
    'Excitement':	['excited', 'enthusiastic', 'energetic', 'playful', 'impatient', 'panicky', 'impulsive', 'hasty'],
    'Fatigue':	['tired', 'sleepy', 'drowsy'],
    'Fear':	['scared', 'fearful', 'timid', 'terrified'],
    'Happiness':	['cheerful', 'delighted', 'happy', 'amused', 'laughing', 'thrilled', 'smiling', 'pleased', 'overwhelmed', 'ecstatic', 'exuberant'],
    'Pain':	['pain'],
    'Peace':	['content', 'relieved', 'relaxed', 'calm', 'quiet', 'satisfied', 'reserved', 'carefree'],
    'Pleasure':	['funny', 'attracted', 'aroused', 'hedonistic', 'pleasant', 'flattered', 'entertaining', 'mesmerized'],
    'Sadness':	['sad', 'melancholy', 'upset', 'disappointed', 'discouraged', 'grumpy', 'crying', 'regretful', 'grief-stricken', 'depressed', 'heartbroken', 'remorseful', 'hopeless', 'pensive', 'miserable'],
    'Sensitivity':	['apologetic', 'nostalgic'],
    'Suffering':	['offended', 'hurt', 'insulted', 'ignorant', 'disturbed', 'abusive', 'offensive'],
    'Surprise':	['surprise', 'surprised', 'shocked', 'amazed', 'startled', 'astonished', 'speechless', 'disbelieving', 'incredulous'],
    'Sympathy':	['kind', 'compassionate', 'supportive', 'sympathetic', 'encouraging', 'thoughtful', 'understanding', 'generous', 'concerned', 'dependable', 'caring', 'forgiving', 'reassuring', 'gentle'],
    'Yearning':	['jealous', 'determined', 'aggressive', 'desperate', 'focused', 'dedicated', 'diligent']
}
CLASS_NAMES = list(EMOTION_LABELS.keys())


logger = logging.getLogger()


def names_to_label(names: list[str]) -> np.ndarray:
    labels = [CLASS_NAMES.index(name) for name in names]
    sparse_label = np.zeros(len(CLASS_NAMES))
    sparse_label[labels] = 1
    return sparse_label


class SampleInfo(TypedDict):
    movie_id: str
    clip_id: int
    emotions: np.ndarray
    start_shot: int
    end_shot: int
    start_frame: int
    end_frame: int
    

# BUG tt0388795 is shorter than expected (only 193256 frames but even no.193401 appears in MG dataset)
class MovieGraphsDataset(Dataset):
    def __init__(
        self,
        data_dir: str = '/ocean/projects/iri180005p/psuzhang/data/MovieGraphs',
        split: Literal['train', 'val', 'test', 'train+val', 'all'] = 'all',
        sampling_strategy: str = 'uniform',
        video_len: int = 8
    ):
        super().__init__()
        # constants
        self.BBOX_TO_MASK_THRESHOLD = 0.5
        self.RESIZE_SIZE = 256
        self.CROP_SIZE = 224
        
        self.data_dir = data_dir
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.video_len = video_len
        
        # by self._create_index()
        self.index: list[SampleInfo]
        self.raw_labels: dict[str, dict[str, list[str]]]
        self.all_scene_boundaries: dict[str, pd.DataFrame]
        self.all_video_boundaries: dict[str, pd.DataFrame]
        self.human_boxes: dict[str, dict[str, list[list[float]]]] # movie_id -> frame_id -> list of boxes
        
        self._create_index()
        
        
    def _create_index(self):
        # load the emotion labels
        label_path = osp.join(self.data_dir, 'official/mg/py3loader/emotion_labels.json')
        with open(label_path, 'r') as f:
            self.raw_labels = orjson.loads(f.read())
        logger.debug(f'Loaded {len(self.raw_labels)} movie labels')
            
        # load the movie list
        split_path = osp.join(self.data_dir, 'official/mg/split.json')
        with open(split_path, 'r') as f:
            split_info = orjson.loads(f.read())
        split_info['train+val'] = split_info['train'] + split_info['val']
        split_info['all'] = split_info['train'] + split_info['val'] + split_info['test'] # for debugging purpose
        logger.debug(f'Loaded {len(split_info[self.split])} movies for {self.split} split')
        
        # load the scene_boundaries (start-shot and end-shot for each sample)
        self.all_scene_boundaries = dict()
        for movie_id in split_info[self.split]:
            fpath = osp.join(self.data_dir, f'official/mg_videoinfo/scene_boundaries/{movie_id}.scenes.gt')
            self.all_scene_boundaries[movie_id] = pd.read_csv(
                fpath, 
                header=None, 
                sep='\s+', # ignore the whitespace at the end of each line
                names=['start_shot', 'end_shot', 'is_useful']
            )
        logger.debug(f'Loaded {len(self.all_scene_boundaries)} scene boundaries')
        
        # load the video_boundaries (start-frame and end-frame for each shot)
        self.all_video_boundaries = dict()
        for movie_id in split_info[self.split]:
            fpath = osp.join(self.data_dir, f'official/mg_videoinfo/video_boundaries/{movie_id}.videvents')
            _start_frames = pd.read_csv(
                fpath,
                skiprows=1,
                header=None,
                sep='\s+', 
                names=['start_frame', 'time_stamp', 'CUT']
            )['start_frame'].values
            _end_frames = (_start_frames-1).tolist() + [-1] # use -1 to indicate the end of the video
            _start_frames = [0] + _start_frames.tolist()
            self.all_video_boundaries[movie_id] = pd.DataFrame({
                'start_frame': _start_frames,
                'end_frame': _end_frames
            })
        logger.debug(f'Loaded {len(self.all_video_boundaries)} video boundaries')
        
        # load the human boxes
        boxes_fpath = osp.join(self.data_dir, 'all_human_boxes.json')
        with open(boxes_fpath, 'r') as f:
            self.human_boxes = orjson.loads(f.read())
        logger.debug(f'Loaded {len(self.human_boxes)} human boxes')
        
        # create the index
        self.index = []
        for movie_id, clips in tqdm(self.raw_labels.items(), desc='Creating index'):
            if movie_id not in split_info[self.split]:
                continue
            for clip_id, emotions in clips.items():
                clip_id = int(clip_id)
                sample_info = {
                    'movie_id': movie_id,
                    'clip_id': clip_id,
                    'emotions': names_to_label(emotions),
                    'start_shot': self.all_scene_boundaries[movie_id].iloc[clip_id]['start_shot'],
                    'end_shot': self.all_scene_boundaries[movie_id].iloc[clip_id]['end_shot'],
                }
                sample_info['start_frame'] = self.all_video_boundaries[movie_id].iloc[sample_info['start_shot']-1]['start_frame']
                sample_info['end_frame'] = self.all_video_boundaries[movie_id].iloc[sample_info['end_shot']-1]['end_frame']
                self.index.append(sample_info)
        logger.info(f'Created index for {self.split} set with {len(self.index)} samples.')
        
        # mannually change the end_frame of the last clip in tt0388795 to avoid out-of-range error
        if self.split == 'train':
            corrupted_clip = self.index[1057]
            assert corrupted_clip['movie_id'] == 'tt0388795'
            assert corrupted_clip['clip_id'] == 207
            corrupted_clip['end_frame'] = 184272 # 02:08:07
        
    
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, i):
        sample_info = self.index[i]
        movie_id = sample_info['movie_id']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        
        # last clip
        if end_frame == -1:
            end_frame = start_frame + 8*24
            
        num_frames = end_frame - start_frame + 1
        # HACK return another sample if the clip is too short
        if self.sampling_strategy != 'uniform' and num_frames < self.video_len:
            new_i = np.random.randint(self.index.shape[0])
            return self.__getitem__(new_i)

        # sparse sampling
        if self.sampling_strategy == 'uniform':
            sampled_frame_indexes = np.linspace(start_frame, end_frame, self.video_len, dtype=int)
        elif self.sampling_strategy == 'random':
            all_frame_indexes = np.arange(start_frame, end_frame+1)
            sampled_frame_indexes = sorted(np.random.choice(all_frame_indexes, size=self.video_len, replace=False))
        
        # load video frames, human bboxes and process them
        frames = []
        masks = []
        for frame_index in sampled_frame_indexes:
            frame_path = osp.join(
                self.data_dir, 
                f'frames/{movie_id}/frame_{frame_index}.jpg'
            )
            raw_frame = Image.open(frame_path).convert('RGB')
            raw_boxes = self.human_boxes[movie_id][f'frame_{frame_index}.jpg']
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
        
        target = torch.from_numpy(sample_info['emotions']).float()
        
        return frames, video_masks, target



if __name__ == '__main__':
    pass