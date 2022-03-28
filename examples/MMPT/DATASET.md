# Dataset

We understand video data are challenging to download and process. For videos, we provide our preprocessing scripts under `scripts/video_feature_extractor` (deeply adapted from `https://github.com/antoine77340/video_feature_extractor`); for text, we pre-tokenizing scripts under `scripts/text_token_extractor`.

### S3D Feature Extraction
We use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction. Please place the models as `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`.

We implement a `PathBuilder` to automatically track video ids, source video paths to their feature locations (you may need `conda install -c anaconda pandas`). Decoding may need `pip install ffmpeg-python`.

### Howto100M
[Howto100M](https://www.di.ens.fr/willow/research/howto100m/) is a large-scale video pre-training datasets. You may download videos by yourself and run preprocessing of our scripts. 

Several key differences of our preprocessing from existing papers: (1) we use `raw_caption.json` instead of `caption.json` to have pure self-supervision on text (`caption.json` has manual removal of stop words); (2) we remove partially duplicated texts that are originally designed for real-time readability (see `mmpt/processors/dedupprocessor.py`); (3) then we shard video/text features using `SharedTensor` in `mmpt/utils/shardedtensor.py` for fast loading during training (faster than `h5py`).

#### Steps
##### video
To extract video features: edit and run `bash scripts/video_feature_extractor/how2/s3d.sh`. (consider to run this on multiple machines; by default, we store features in fp16 to save space and also for faster training).

Split available video ids as `data/how2/how2_s3d_train.lst` and `data/how2/how2_s3d_val.lst`.

Lastly, pack video features into `ShardedTensor` using `python scripts/video_feature_extractor/shard_feature.py`.

##### text
Clean captions using `python -m mmpt.processors.dedupprocessor`.

Tokenize dedupped captions `data/how2/raw_caption_dedup.pkl` into sharded numpy arrays:  
```
python scripts/text_token_extractor/pretokenization.py scripts/text_token_extractor/configs/bert-base-uncased.yaml
```

### Youcook, MSRVTT etc.
We use the version of Youcook and MSRVTT come with Howto100M and MILNCE. Please download the data to `data/youcook` and `data/msrvtt` accordingly, you can also check `projects/task/youcook.yaml` and `projects/task/vtt.yaml` etc. in details. 
We extract features for Youcook, MSRVTT similar to the first step of Howto100M but we read text from meta data directly and perform on-the-fly tokenization.

