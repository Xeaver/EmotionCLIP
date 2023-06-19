import json
import numpy as np
import re
import os
from deepmultilingualpunctuation import PunctuationModel
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='youtube',
                    help='the dir to save the dataset')
# parser.add_argument('--s3_address', type=str,
#                     help='address/ access point of a s3 bucket followed by dir name')
# parser.add_argument('--fps', type=int, default=8,
#                     help='output fps for videos')
# parser.add_argument('--n_workers', type=int, default=12,
#                     help='output fps for videos')
args = parser.parse_args()

def get_word_time_range(model,transcript,f_path):
    """
        model: PunctuationModel
            model.predict returns value in the form of [word,punc,punc_confidence]
        transcript: youtube transcript in the form of [{text,start,duration}]
        
        return:
        out: [{word,punc,punc_conf,time_range:(start,end)}]
    """
    status = {'same_length':True}
    reformate_transcript = list()
    n = len(transcript)
    for i,t in enumerate(transcript):
        if i==n-1:
            end = t['start']+t['duration']
        else:
            end = min(t['start']+t['duration'],transcript[i+1]['start'])
        start = t['start']
        texts = model.preprocess(t['text'])
        if len(texts) == 0:
            continue
        step_size = (end-start)/len(texts)
        word_time_step = [(start+i*step_size,start+(i+1)*step_size) for i in range(len(texts))]
        reformate_transcript.append({'text':texts,'start':start,'end':end,'word_sec':word_time_step})
    clean_text =  [r for t in reformate_transcript for r in t['text']]
    time_range = [r for t in reformate_transcript for r in t['word_sec']]
    # clean_text = model.preprocess(text)
    labled_words = model.predict(clean_text)
    if len(labled_words) != len(time_range):
        status = {'same_length':True}
        print(f_path,len(labled_words),len(time_range))
    # assert len(labled_words) == len(time_range)
    out = [{'word':labled_words[i][0],'punc':labled_words[i][1],'punc_conf':labled_words[i][2],'time_range':time_range[i]} for i in range(len(labled_words))]
    return out, status

def regroup_by_punc(word_punc_with_time,split_punc='.',exclude_on=None):
    """
        word_punc_with_time: return value from "get_word_time_range()"
        split_punc: the punc to split the transcript into sentences
        exlcude_on: regex to exclude certain sentences
        
        return:
        sentence_range:[{text,start,end,duration}]
    """
    split_indices = [i for i,w in enumerate(word_punc_with_time) if w['punc']==split_punc]
    regrouped = [word_punc_with_time[split_indices[i]+1:split_indices[i+1]+1] for i in range(len(split_indices[:-1]))]
    sentence_range = [{'text':' '.join([w['word'] for w in t]),'start':min([t[ti]['time_range'][0] for ti in range(len(t))]),'end':max([t[ti]['time_range'][1] for ti in range(len(t))]),'duration':max([t[ti]['time_range'][1] for ti in range(len(t))])-min([t[ti]['time_range'][0] for ti in range(len(t))])} for t in regrouped]
    if exclude_on is not None:
        sentence_range = [s for s in sentence_range if not re.match(exclude_on, s['text'])]
    return sentence_range

if __name__=='__main__':
    total_ne = 0
    total_processed =0
    model = PunctuationModel()
    playlist_movies = [os.path.join(args.save_dir,d) for d in os.listdir(args.save_dir)]
    for video_dir in tqdm(playlist_movies):
        with open(os.path.join(video_dir,'transcript.json'), 'r') as f:
            transcript = json.load(f)
        cleaned_transcript_path = os.path.join(video_dir,'transcript_cleaned.json')
        if not os.path.exists(cleaned_transcript_path) or True:
            total_processed+=1
            word_punc_with_time,status = get_word_time_range(model,transcript,cleaned_transcript_path)
            cleaned_text = regroup_by_punc(word_punc_with_time,split_punc='.',exclude_on='.*\[')
            if status['same_length']:
                with open(cleaned_transcript_path, 'w') as f:
                    json.dump(cleaned_text,f)
            else:
                total_ne+=1
    print(total_ne/total_processed)