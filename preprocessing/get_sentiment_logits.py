import argparse
import os
import os.path as osp

from tqdm import tqdm
import orjson
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', type=str, default='/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_raw_index.json')
    parser.add_argument('--save-path', type=str, default='/ocean/projects/iri180005p/psuzhang/projects/Movie-CLIP/data/youtube/tv_sentiment_logits.npy')
    parser.add_argument('--pretrained-sentiment-model-name', type=str, default='j-hartmann/emotion-english-distilroberta-base')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    # load index
    with open(args.index_path, 'r') as f:
        index = orjson.loads(f.read())
    print(f'Loaded index with {len(index)} entries')

    # load pretrained sentiment model
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # disable tokenizers parallelism
    sentiment_pipeline = transformers.pipeline('text-classification', model=args.pretrained_sentiment_model_name)
    tokenizer = sentiment_pipeline.tokenizer
    model = sentiment_pipeline.model.to('cuda').eval()
    print(f'Loaded sentiment model {args.pretrained_sentiment_model_name}')

    # get logits
    all_logits = []
    for i in tqdm(range(0, len(index), args.batch_size)):
        batch = index[i:i+args.batch_size]
        texts = [x['text'] for x in batch]
        with torch.no_grad():
            tokenized = tokenizer(texts, padding='max_length', max_length=77, truncation=True)
            tokenized_text = torch.tensor(tokenized['input_ids'], dtype=torch.long).cuda()
            tokenized_attn_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long).cuda()
            logits = model(tokenized_text, attention_mask=tokenized_attn_mask, return_dict=True).logits
            logits = logits.cpu().numpy() # shape = [batch_size, n_classes]
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits, axis=0) # shape = [len(index), n_classes]
    print(f'Got logits with shape {all_logits.shape}')

    # save to .npy
    with open(args.save_path, 'wb') as f:
        np.save(f, all_logits)
    print(f'Saved logits to {args.save_path}')