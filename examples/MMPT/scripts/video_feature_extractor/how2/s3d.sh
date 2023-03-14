#!/bin/bash


# python scripts/video_feature_extractor/extract.py \
#     --vdir <path_to_video_folder> \
#     --fdir data/feat/feat_how2_s3d \
#     --type=s3d --num_decoding_thread=4 \
#     --batch_size 32 --half_precision 1

# python scripts/video_feature_extractor/extract.py \
#     --vdir /shares/volk.cl.uzh/zifjia/YouCookII/raw_videos_tiny/validation \
#     --fdir data/feat/feat_youcook_s3d \
#     --type=s3d --num_decoding_thread=0 \
#     --batch_size 1 --half_precision 1

python scripts/video_feature_extractor/extract.py \
    --vdir /shares/volk.cl.uzh/zifjia/RWTH_Fingerspelling/video \
    --fdir data/feat/feat_rwthfs_s3d \
    --type=s3d --num_decoding_thread=2 \
    --batch_size 8 --half_precision 1
