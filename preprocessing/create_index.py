import argparse
import os
import os.path as osp
import json
import math
import sys
import tempfile
from typing import Optional

import h5py
from tqdm import tqdm
from rich import print as rprint

BOLD_DUPLICATE_ID = [
    '7YpF6DntOYw',
    'Db19rWN5BGo',
    '_a9SWtcaNj8',
    'lWXhqIAvarw',
    'CrlfWnsS7ac',
    'y7ncweROe9U',
    'CrlfWnsS7ac',
    'phrYEKv0rmw',
    'oD_wxyTHJ2I',
    'CrlfWnsS7ac',
    'y7ncweROe9U',
    'oD_wxyTHJ2I',
    '914yZXz-iRs',
    '2qQs3Y9OJX0',
    'phrYEKv0rmw',
    'phrYEKv0rmw',
    '_mAfwH6i90E',
    'phrYEKv0rmw',
    'CrlfWnsS7ac'
]

def is_path_writable(path: str) -> bool:
    abs_path = osp.abspath(osp.expandvars(osp.expanduser(path)))
    try:
        with tempfile.TemporaryFile(dir=osp.dirname(abs_path)):
            pass
        return True
    except OSError:
        return False


# create an index for all valid samples
def create_index(
    video_path: str, 
    caption_path: str,
    neutral_score_path: str,
    save_path: Optional[str] = None,
    fps: int = 8,
    video_len: int = 8,
):
    assert osp.isdir(video_path), f'{video_path} does not exist'
    assert osp.isdir(caption_path), f'{caption_path} does not exist'
    assert osp.isfile(neutral_score_path), f'{neutral_score_path} does not exist'
    if save_path:
        assert is_path_writable(save_path), f'{save_path} is not writable'
    
    valid_video_ids = []
    index = []
    
    # check that caption file, video file, and human_box file all exist
    all_video_ids = set(os.listdir(caption_path)) | set(os.listdir(video_path))
    for video_id in tqdm(all_video_ids, desc='Check files', leave=True):
        # skip BoLD duplicate videos
        if video_id.split('&')[0] in BOLD_DUPLICATE_ID:
            continue
        caption_file = osp.join(caption_path, video_id, 'transcript_cleaned.json')
        video_file = osp.join(video_path, video_id, 'frames.hdf5')
        human_box_file = osp.join(video_path, video_id, 'human_boxes.json')
        if osp.isfile(caption_file) and osp.isfile(video_file) and osp.isfile(human_box_file):
            valid_video_ids.append(video_id)

    rprint(f'Found {len(valid_video_ids)} available videos.')
    
    # load neutral_scores
    with open(neutral_score_path, 'r') as f:
        neutral_scores = json.load(f)
    
    # create index and save human_boxes
    for video_id in tqdm(valid_video_ids, desc='Creating index', leave=True):
        # load caption
        caption_file = osp.join(caption_path, video_id, 'transcript_cleaned.json')
        with open(caption_file, mode='rb') as f:
            caption_data = json.load(f)
        
        # load human_boxes
        human_box_file = osp.join(video_path, video_id, 'human_boxes.json')
        with open(human_box_file, mode='rb') as f:
            human_box_data: list[dict] = json.load(f)
        # discard frames with no human box
        human_box_data: dict[str, list] = {k: v for sample_dict in human_box_data for k, v in sample_dict.items() if len(v)}
        frames_with_human_box = list(map(lambda s: int(s.split('.')[0]), human_box_data.keys())) # example key: '00001321.jpg'
        
        # load video frame keys
        with h5py.File(osp.join(video_path, video_id, 'frames.hdf5'), mode='r') as f:
            frames_in_hdf5 = list(f.keys())
        frames_in_hdf5 = set(map(lambda s: int(s.split('.')[0]), frames_in_hdf5))
        
        
        # filter samples with multiple criteria 
        for i, caption in enumerate(caption_data):
            start_frame_idx = math.ceil(caption['start'] * fps)
            end_frame_idx = math.floor(caption['end'] * fps)
            frames_to_be_used = set(range(start_frame_idx, end_frame_idx + 1))
            
            if len(frames_to_be_used) >= video_len and frames_to_be_used.issubset(frames_in_hdf5):
                has_person = len(frames_to_be_used.intersection(frames_with_human_box)) / len(frames_to_be_used)
                sample = {
                    'video_id': video_id,
                    'duration': caption['duration'],
                    'start_frame': start_frame_idx,
                    'end_frame': end_frame_idx,
                    'text': caption['text'],
                    'neutral_score': neutral_scores[video_id][i],
                    'has_person': has_person,
                }
                index.append(sample)
                
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(index, f, indent=4)
    
    n_samples = len(index)
    n_samples_with_person_all = len(list(filter(lambda s: s['has_person'] == 1, index)))
    n_samples_with_person_partial = len(list(filter(lambda s: s['has_person'] > 0., index)))
    rprint(f'{n_samples=}')
    rprint(f'{n_samples_with_person_partial=}')
    rprint(f'{n_samples_with_person_all=}')
    
    return index
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default='/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_index.json')
    parser.add_argument('--video-path', type=str, default='/ocean/projects/iri180005p/shared/video-data/video-sub/youtube')
    parser.add_argument('--caption-path', type=str, default='/ocean/projects/iri180005p/shared/tv_cleaned')
    parser.add_argument('--neutral-score-path', type=str, default='/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_neutral_scores.json')
    args = parser.parse_args()

    create_index(
        video_path=args.video_path,
        caption_path=args.caption_path,
        neutral_score_path=args.neutral_score_path,
        save_path=args.save_path,
    )