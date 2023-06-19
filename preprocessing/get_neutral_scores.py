import argparse
import os
import os.path as osp
from types import SimpleNamespace
import sys
import json

import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import transformers
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, default='/ocean/projects/iri180005p/shared/video-data/video-sub/youtube')
    parser.add_argument('--caption-path', type=str, default='/ocean/projects/iri180005p/shared/tv_cleaned')
    parser.add_argument('--save-path', type=str, default='/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_neutral_scores.json')
    parser.add_argument('--pretrained-sentiment-model-name', type=str, default='j-hartmann/emotion-english-distilroberta-base')
    args = parser.parse_args()

    # check that caption file, video file, and human_box file all exist
    valid_video_ids = []
    all_video_ids = set(os.listdir(args.caption_path)) | set(os.listdir(args.video_path))
    for video_id in tqdm(all_video_ids, desc='Check files', leave=True):
        if video_id.split('&')[0] in BOLD_DUPLICATE_ID:
            continue
        caption_file = osp.join(args.caption_path, video_id, 'transcript_cleaned.json')
        video_file = osp.join(args.video_path, video_id, 'frames.hdf5')
        # human_box_file = osp.join(video_path, video_id, 'human_boxes.json')
        if osp.isfile(caption_file) and osp.isfile(video_file):
            valid_video_ids.append(video_id)
    
    
    failed = 0
    sentiment_score = {}
    classifier = transformers.pipeline("text-classification", model=args.pretrained_sentiment_model_name, return_all_scores=True)
    for video_id in tqdm(valid_video_ids):
        caption_file = osp.join(args.caption_path, video_id, 'transcript_cleaned.json')
        if not osp.isfile(caption_file):
            rprint(video_id)
            failed += 1
            continue
        with open(caption_file, 'r') as f:
            data = json.load(f)
        all_texts = [sample['text'] for sample in data]
        results = classifier(all_texts)
        neutral_scores = [pred['score'] for r in results for pred in r if pred['label'] == 'neutral']
        sentiment_score[video_id] = neutral_scores

    with open(args.save_path, 'w') as f:
        json.dump(sentiment_score, f, indent=4)

    rprint(f'{failed} videos failed')
    rprint(f'{len(sentiment_score)} videos processed')