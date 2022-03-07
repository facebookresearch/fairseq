#!/bin/bash


python scripts/video_feature_extractor/extract.py \
    --vdir <path_to_video_folder> \
    --fdir data/feat/feat_how2_s3d \
    --type=s3d --num_decoding_thread=4 \
    --batch_size 32 --half_precision 1
