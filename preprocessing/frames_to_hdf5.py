import json
import numpy as np
import re
import os
import multiprocessing as mp
import argparse
from tqdm import tqdm
import h5py
import tarfile
from functools import partial

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='youtube',
                    help='the dir to save the dataset')
parser.add_argument('--out_dir', type=str, default='youtube',
                    help='the dir to save the hdf5 file')
parser.add_argument('--n_workers', type=int, default=12,
                    help='number of workers')
parser.add_argument('--frames_file', type=str, default='frames',
                    help='read from images if "frames" else read from "frames_8_fps.tar.gz"')
args = parser.parse_args()


def tar_gz_to_hdf5(vid_name,out_dir):
    base_path = os.path.split(vid_name)[-1]
    save_path = os.path.join(out_dir,base_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(os.path.join(save_path,'frames.hdf5')):
        return True
    frames_tar = os.path.join(vid_name, 'frames_8_fps.tar.gz')
    if not os.path.exists(frames_tar):
        return False
    hf = h5py.File(os.path.join(save_path,'frames.hdf5'), 'a')

    with tarfile.open(frames_tar,'r:gz') as tarobj:
        for mem in tarobj.getnames()[1:]:
            fname = mem.split('/')[-1]
            img_bytes = tarobj.extractfile(mem).read()
            hf[fname]=np.array(img_bytes)

    hf.close()
    os.remove(frames_tar)
    return True

def frames_to_hdf5(vid_name,out_dir):
    base_path = os.path.split(vid_name)[-1]
    save_path = os.path.join(out_dir,base_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(os.path.join(save_path,'frames.hdf5')):
        return True
    frames_dir = os.path.join(vid_name, 'frames')
    if not os.path.exists(frames_dir):
        return False
    hf = h5py.File(os.path.join(save_path,'frames.hdf5'), 'a')

    for k in os.listdir(frames_dir):   # find all images inside a.
        img_path = os.path.join(frames_dir, k)
        with open(img_path, 'rb') as img_f:  # open images as python binary
            binary_data = img_f.read()

        binary_data_np = np.asarray(binary_data)
        hf[k] = binary_data_np 

    hf.close()
    return True


def update_progress_bar(_):
    progress_bar.update()

if __name__ == '__main__':
    playlist_movies = os.listdir(args.save_dir)
    print(len(playlist_movies))
    global progress_bar
    progress_bar = tqdm(total=len(playlist_movies))
    pool = mp.Pool(args.n_workers)
    for frames_dir in playlist_movies:
        if args.frames_file == 'frames':
            pool.apply_async(partial(frames_to_hdf5,out_dir=args.out_dir), (os.path.join(args.save_dir,frames_dir),), callback=update_progress_bar)
        elif 'tar.gz' in args.frames_file:
            pool.apply_async(partial(tar_gz_to_hdf5,out_dir=args.out_dir), (os.path.join(args.save_dir,frames_dir),), callback=update_progress_bar)
    pool.close()
    pool.join()