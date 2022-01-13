# Pretraining

(If you are new to the ideas of `mmpt.processors`, see [README](README.md) first.)
We mostly use [howto100M](https://github.com/antoine77340/howto100m) dataset for pretraining (other datasets are coming). So you are less likely to write a new `MetaProcessor`, `VideoProcessor` or `TextProcessor` but only working on a new `Aligner`, a new model and loss.

### Data Sharding
Pretraining on Howto100M is heavy on IO since we have millions of videos or captions on the hard disk that cannot be fit into the memory. 
It is desirable to have an optimized preprocessing step before the actual dataloading.  

We support data sharding to pack multiple videos into a shards of training data for both videos and captions. (see [dataset](DATASET.md) for preprocessing).
These shards will be mapped into memory to reduce the frequency of IO access on millions of files. See (processors starting with `Sharded*`).
This will be the default config for a how2 dataset `projects/task/how2.yaml`.

Great thanks to Dmytro Okhonko for sharing the code from MARGE project.

### Training
Pretraining on Howto100m is expected on one or multiple nodes, where each node has 8 GPUS with 32 GB mem.
launching a pretraing on MFM+MLM can be done, via:  
```python locallaunch.py projects/mfmmlm/how2.yaml```

### Pre-training with a Retrieval Model (VideoCLIP)
This projects now support alternatively run a retrieval model and pre-training.
We implement a basic retrieval model that is built on the hidden states of a video and faiss.

You may need to install faiss via `conda install faiss-cpu -c pytorch`.  

Right now, the hidden states of a video is computed as the average of 8 clips of their pooled visual/text hidden states.
See `mmpt/tasks/retritask.py` for more details.
The `.yaml` config for running pre-training with a retrieval model can be found at `projects/retri/videoretri.yaml`.
