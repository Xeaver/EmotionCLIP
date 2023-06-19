from __future__ import annotations
import os
import os.path as osp
from typing import Callable, Optional, Literal, TypedDict
import math
import json
import logging

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import orjson

BLACK_LIST = ['.DS_Store']

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
    start_frame: int
    end_frame: int
    label_frame: int

# BUG tt0388795 is shorter than expected (only 193256 frames but even no.193401 appears in MG dataset)
class LirisAccede(Dataset):
    def __init__(
        self,
        data_dir: str = '/ocean/projects/iri180005p/psuzhang/data/LIRIS-ACCEDE/MediaEval2018/',
        split: Literal['train', 'test'] = 'train',
        sampling_strategy: str = 'uniform',
        video_len: int = 8,
        preload_human_boxes: bool = True,
        
    ):
        self.data_dir = data_dir
        self.split = split
        self.sampling_strategy = sampling_strategy
        self.video_len = video_len
        self.preload_human_boxes = preload_human_boxes
        
        # by self._create_index()
        self.index: list[SampleInfo]
        self.human_boxes: dict[str, dict[str, list[list[float]]]]
        self.all_annotations: dict[str, pd.DataFrame]
        # self.all_video_boundaries: dict[str, pd.DataFrame]
        
        self._create_index()
        
        # clip-style preprocesser
        if split == 'train':
            self.preprocesser = T.Compose([
                T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocesser = T.Compose([
                T.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def _create_start_end_frames(self,video_name, total_frames):
        index = []
        # total_frames = total_frames - 1
        for i in range(total_frames): 
            if i < np.ceil(self.video_len/2):
                if self.split == 'train': # filter out the begining of videos
                    continue
                sample_info = {
                        'movie_id': video_name,
                        'start_frame': 0,
                        'end_frame': self.video_len, # exclusive
                        'label_frame': i,
                        }
            elif (total_frames-i) < np.floor(self.video_len/2): 
                if self.split == 'train': # filter out the end of videos
                    continue
                sample_info = {
                        'movie_id': video_name,
                        'start_frame': total_frames-self.video_len,
                        'end_frame': total_frames, # exclusive
                        'label_frame': i-(total_frames-self.video_len)
                        }
            else:
                sample_info = {
                        'movie_id': video_name,
                        'start_frame': int(i-np.ceil(self.video_len/2)),
                        'end_frame': int(i+np.floor(self.video_len/2)), # exclusive
                        'label_frame': int(np.ceil(self.video_len/2))
                        }
            
            has_all_frames = True
            for i in range(sample_info['start_frame'],sample_info['end_frame']):
                frame_file = osp.join(
                        self.data_dir, 
                        f'frames/{video_name}.mp4',
                        self._idx_to_name(i)
                    )
                if not osp.exists(frame_file):
                    has_all_frames = False
                    print(f'sample {frame_file} does not exist')
                    break
            if has_all_frames:
                index.append(sample_info)
        return index
    
    def _create_index(self):
        # load the emotion labels
        self.all_annotations = {}
        self.index = []
        self.human_boxes = {}
        label_path = osp.join(self.data_dir, 'annotations/')
        video_path = osp.join(self.data_dir, 'frames/')
        if self.split == 'train':
            label_path_list = [p for p in os.listdir(label_path) if 'test' not in p.lower()]
        else:
            label_path_list = [p for p in os.listdir(label_path) if 'test' in p.lower()]
        for label_dir in label_path_list:
            abs_label_dir = osp.join(label_path,label_dir,'annotations/')
            for label_file in os.listdir(abs_label_dir):
                if label_file in BLACK_LIST: continue
                abs_label_file = osp.join(abs_label_dir,label_file)
                video_name = "_".join(label_file.split('_')[:2])
                self.all_annotations[video_name] = pd.read_csv(abs_label_file,delimiter='\t')
                if self.all_annotations[video_name].isnull().values.any():
                    print("has_na:",video_name)
                frames_dir = osp.join(
                        self.data_dir, 
                        f'frames/{video_name}.mp4'
                    )
                self.index+=self._create_start_end_frames(video_name, total_frames=min([len(os.listdir(frames_dir)),self.all_annotations[video_name].shape[0]]))
        # check frames numbers
        for sample_info in self.index:
            movie_id = sample_info['movie_id']
            start_frame = sample_info['start_frame']
            end_frame = sample_info['end_frame']
            target_frame = sample_info['label_frame']
            if end_frame - start_frame != 8 or target_frame < 0 or target_frame > 7 or end_frame > self.all_annotations[movie_id].shape[0]:
                print(sample_info,self.all_annotations[video_name].Time)
        print('Checking complete')
        # preload all human box annotations to speed up the __getitem__ function
        if self.preload_human_boxes:
            for video_id in tqdm(self.all_annotations.keys(), desc='Loading human box annotations'):
                human_box_path = osp.join(self.data_dir, 'bboxes', f"{video_id}.mp4", 'human_boxes.json')
                with open(human_box_path, mode='rb') as f:
                    _human_box_data: list[dict[str, list[list[float]]]] = orjson.loads(f.read())
                human_box_data: dict[str, list[list[float]]] = {k: v for sample_dict in _human_box_data for k, v in sample_dict.items()}
                self.human_boxes[video_id] = human_box_data
    
    def _idx_to_name(self, n: int, width: int = 8) -> str:
        file_name = str(n+1).zfill(width) + '.jpg'
        return file_name
    
    
    # Scale the bboxes accordingly, since the original bboxes are extracted from the scaled images 
    def _get_correct_bbox(
        self, 
        boxes: list[list[float]],
        img_shape: tuple[int, int],
        yolo_shape: tuple[int, int] = (640, 640)
    ) -> list[list[float]]:
        r = min(yolo_shape[0] / img_shape[0], yolo_shape[1] / img_shape[1])
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        RESIZE_SIZE = 256
        CROP_SIZE = 224
        bboxes = [self._get_correct_bbox(b, f.size) for f, b in zip(frames, all_boxes)]
        processed_frames = []
        processed_masks = []
        for frame, bbox in zip(frames, bboxes):
            mask = self._bbox_to_mask(bbox, frame.size)
            resized_frame = F.resize(frame, size=RESIZE_SIZE, interpolation=T.InterpolationMode.BICUBIC)
            resized_mask = F.resize(mask.unsqueeze(0), size=RESIZE_SIZE).squeeze()
            cropped_frame = F.center_crop(resized_frame, output_size=CROP_SIZE)
            cropped_mask = F.center_crop(resized_mask, output_size=CROP_SIZE)
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
        
    
    def __len__(self):
        return len(self.index)
    
    
    def __getitem__(self, i):
        sample_info = self.index[i]
        movie_id = sample_info['movie_id']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        target_frame = sample_info['label_frame']
        
        # load video frames
        frames = []
        bboxes = []
        for frame_index in range(start_frame,end_frame):
            frame_path = osp.join(
                self.data_dir, 
                f'frames/{movie_id}.mp4/{self._idx_to_name(frame_index)}'
            )
            frame = Image.open(frame_path).convert('RGB')
            bbox = self.human_boxes[movie_id][self._idx_to_name(frame_index)]
            frames.append(frame)
            bboxes.append(bbox)
        video, video_mask = self._preprocess_frames(frames, bboxes)
        
        target = torch.from_numpy(np.squeeze(self.all_annotations[movie_id].loc[self.all_annotations[movie_id]['Time'] == (start_frame+target_frame)].values[:,1:])).float()
        video = video[target_frame].repeat(self.video_len, 1, 1, 1)
        video_mask = video_mask[target_frame].repeat(self.video_len, 1, 1)
        return video, video_mask, target



if __name__ == '__main__':
    pass