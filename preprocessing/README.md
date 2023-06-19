# The data collection and preprocessing pipeline for "Learning Emotion Representations from Verbal and Nonverbal Communication".

## Setup

### To setup human detection part, please run the following code and follow their setup.

```
git clone https://github.com/WongKinYiu/yolov7.git
cp yolo_additions/datasets.py yolov7/utils/datasets.py
cp yolo_additions/detect_human.py yolov7/detect_human.py
```

### After setting up yolov7, please run the following installation for YouTube API usage.

```
pip install youtube_transcript_api
pip install pytube
pip install youtube-search-python
```

### To setup the punctuation model to regroup the transcripts, run the following code:

```
pip install deepmultilingualpunctuation
```

### Then, you may need [ffmpeg](https://ffmpeg.org/) for video processing.

## Run

We use the following default setting but you may change them as needed.

1. You can seach for videos on YouTube using:

   ```
   python search_youtube.py
   ```

   You can optionally find the list used in the paper [here](https://drive.google.com/file/d/1Uk7KQLvAo041he-TOhWDiyI7i-ufw-Im/view?usp=sharing).

2. Download the videos using:

   ```
   python download_videos.py --source 'youtube' --fps 8
   ```

   Note: there is a bug in the dependency which is noted in `download_videos.py`.

   You can optionally store all the frames in hdf5 format using:

   ```
   python frames_to_hdf5.py
   ```

3. Generate human bounding boxes using:

   ```
   cd yolov7
   python detect_human.py --source '../youtube'
   ```

4. Regroup the transcript using:

   ```
   python regroup_transcripts.py
   ```

   Note: if you get "You need to have sentencepiece installed to convert a slow tokenizer to a fast one.", you can fix it using:

   ```
   pip install sentencepiece
   ```

5. To speed up the training, you may optionally pre-compute the emotion scores and logits of all captions using:

   ```
   python get_neutral_scores.py
   python get_sentiment_logits.py
   ```

6. Create the index for training using:

   ```
   python create_index.py \
      --save-path <path_to_save_the_index> \
      --video-path <path_to_the_video_folder> \
      --caption-path <path_to_the_caption_folder> \
      --neutral-score-path <path_to_the_neutral_score_file>
   ```