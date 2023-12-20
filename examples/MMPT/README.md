# SignCLIP - Connecting text and sign language by contrastive learning

This codebase is an adaption of [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT), where general videos (e.g., [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)) are replaced by specific sign language videos (e.g., [How2Sign](https://how2sign.github.io/)) to bring together text and sign language under a same latent space. See VideoCLIP's original README for an overall introduction of multimodal video understanding and instructions on how to install and use the packages.

Video is the most available and rawest representational format that contains human motion and sign language. However, videos are very dense both temporally (*FPS*, frame per second) and spatially (resolution), which are not computataionally efficient and thus require a video encoder to extract informative features with reduced dimentionalties for downstream tasks. 

VideoCLIP uses a [S3D](https://github.com/antoine77340/S3D_HowTo100M) model pretrained on the HowTo100M instructional videos as the video encoder and it produces one video token per second. For sign language, it is possible to use a similar video encoder pretrained on sign language videos. A prominent one is the [I3D](https://www.robots.ox.ac.uk/~vgg/research/bslattend/) model pretrained specifically on the sign language recognition task of British Sign Language.

A potentially more interpretable and universal way of extracting sign language related features from videos is human pose estimation, for example by [MediaPipe Holistic](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md). Recently, quantization-based approaches (such as [SignVQNet](http://nlpcl.kaist.ac.kr/~projects/signvqnet)) have also appeared as an alternative to convert continuous representations of sign language (videos/poses) to discreate tokens similar to spoken language (sub-)words.

Given a 10-second 480p (640×480) RGB (3 channels) video with 30 FPS, we make a rough comprison between the dimentionalities of different video encoders:

| Encoder | Temporal dim. | Spatial dim. | 
|-----------|-----------|-----------|
| Original video | 10x30 | 640×480x3 |
| S3D (pretrained on HowTo100M) | 10 | 512 |
| I3D (pretrained on BSL) | 10 | 1024 |
| MediaPipe Holistic | 10x30 | 543 |
| SignVQNet | 10 | 1024 |

On the text side, we follow VideoCLIP and use the pretrained [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model. One additional idea is to use [SignWriting](https://github.com/sign-language-processing/transcription/blob/aa2b1ead7d39b2d545b83bac2041b4b539471a7c/pose_to_text/IDEA-CLIP.md) as a phonetic text representation of sign language.

## Fingerspelling

We start with a simple dataset, [RWTH German Fingerspelling Database](https://www-i6.informatik.rwth-aachen.de/aslr/fingerspelling.php), which contains 35 gestures with video sequences for the signs A to Z and SCH, the German umlauts Ä, Ö, Ü, and for the numbers 1 to 5. Five of the gestures contain inherent motion (J, Z, Ä, Ö and Ü).

The database consists of 1400 videos that contain gestures of 20 different persons. The gestures were recorded by two different cameras, one webcam and one camcorder, from two different points of view. The webcamc named `cam1` recorded the dominant hands only with a resolution of 320x240 at 25 frames per second, and the camcorder named `cam2` recorded the whole body with a resolution of 352x288 at 25 frames per second. We exclude all `cam1` videos for pose estimation since we assume that the pose estimation model expects whole body input.

Each video file is named `<signer_id>_<letter_id>_<seq_id>_<camera_id>.mpg`. We take all video files and split them randomly into training, dev, and test set at the ratio 8:1:1.

### Training

For contrastive training, we use a batch size of 35 and make each batch a collection of 35 different gestures with their corresponding text prompt `'Fingerspell the letter <letter_name> in German Sign Language.'` regardless of the cemara type. By optimizing the [InfoNCE loss](https://arxiv.org/abs/1807.03748), we want the video embedding of a gesture to move closer to the corresponding text embedding, as well as to move further away to the rest 34 negative examples in the same batch.

We test the viability of the idea using different combinations of video encoders/features:

- [S3D](https://github.com/antoine77340/S3D_HowTo100M) HowTo100M video features
- [I3D](https://www.robots.ox.ac.uk/~vgg/research/bslattend/data/bsl5k.pth.tar) BSL video features ([code](https://github.com/gulvarol/bsl1k))
    - the original 1024 dimentionality
    - the S3D 512 dimentionality downsampled by average pooling
- Mediapipe pose estimation ([code](https://github.com/J22Melody/pose-pipelines)) 
    - (all poses are 2D normalized and the legs are removed)
    - full body
    - dominant hand only
    - 3D hand normalization
    - 2D pose augmentation (training only)


and training strategies:

- (zero-shot VideoCLIP, no training)
- fine-tune VideoCLIP
- from scratch

The same as the original VideoCLIP model training, in all experiments, the video encoders are frozen and only the weights of the two separate
trainable Transformers (one each for video and text) are updated, initilized with the pre-trained `bert-base-uncased`. We train 25 epochs from scratch and 10 epochs for finetuning, respectively.

### Evaluation

We evaluate the models by viewing fingerspelling understanding as a text-video/video-text retrieval task. For both directions, the candidates are ranked by the cosine similarity to the text/video query in the latent space.

For each test text prompt `'Fingerspell the letter <letter_name> in German Sign Language.'`, there is possibly more than one correct videos (e.g, the same letter signed by different signers) in the test video pool, and they are all considered a successful retrieval. We therefore evaluate the text-video retrieval task by `precision@k`, i.e., in the k most similar candidates, how many of them are actual correct answers.

For each test video example, there is only one correct text prompt out of the 35 possible prompts. We therefore evaluate the video-text retrieval task by `recall@k`, i.e., by taking the k most similar candidates, how much is the chance that one of them is the correct answer.

When `k=1`, both `precision@k` and `recall@k` can be interpreted as the retrieval accuracy. For both directions we add an additional metric `Median R`, which is the median value of the index of the first correct answer in the candidate lists. Here are the evaluation results:

|                            | T2V_P@1 | T2V_P@5 | T2V_P@10 | T2V_Median R | V2T_R@1 | V2T_R@5 | V2T_R@10 | V2T_Median R | notes                                                      |
|----------------------------|---------|---------|----------|--------------|---------|---------|----------|--------------|------------------------------------------------------------|
| test_rwthfs_zs             | 0.0286  | 0.0171  | 0.0286   | 22           | 0.0231  | 0.1386  | 0.2772   | 18           | zero-shot VideoCLIP (S3D HowTo100M video feature)           |
| test_rwthfs_videoclip      | 0.4000  | 0.3600  | 0.3000   | 2            | 0.3135  | 0.7459  | 0.8944   | 2            | fine-tune VideoCLIP (S3D HowTo100M video feature)          |
| test_rwthfs_scratch        | 0.5429  | 0.3543  | 0.2829   | 1            | 0.2772  | 0.6898  | 0.8680   | 3            | train from scratch (S3D HowTo100M video feature)           |
| test_rwthfs_scratch_i3d_512| 0.7429  | 0.5600  | 0.4429   | 1            | 0.4653  | 0.8185  | 0.9406   | 2            | train from scratch (I3D BSL-1K video feature, downsampled from 1024 to 512) |
| test_rwthfs_scratch_i3d    | 0.6286  | 0.4743  | 0.3743   | 1            | 0.3729  | 0.7756  | 0.9142   | 2            | train from scratch (I3D BSL-1K video feature)               |
| test_rwthfs_scratch_pose   | 0.8857  | 0.6686  | 0.4171   | 1            | 0.6797  | 0.9739  | 1.0000   | 1            | train from scratch (pose full body feature)                |
| test_rwthfs_scratch_pose_aug| 0.8286  | 0.6400  | 0.4143   | 1            | 0.6732  | 0.9804  | 1.0000   | 1            | train from scratch (pose full body feature with 2D aug.)  |
| test_rwthfs_scratch_hand_dominant | 1.0000  | 0.7200  | 0.4200   | 1            | 0.8170  | 0.9869  | 1.0000   | 1            | train from scratch (pose dominant hand feature)           |
| test_rwthfs_scratch_hand_dominant_norm| 0.8000  | 0.5543  | 0.3429   | 1            | 0.6078  | 0.8889  | 0.9346   | 1            | train from scratch (pose dominant hand feature with 3D norm.) |
| test_rwthfs_scratch_hand_dominant_aug | 0.9143  | 0.7429  | 0.4257   | 1            | 0.9346  | 1.0000  | 1.0000   | 1            | train from scratch (pose dominant hand feature with 2D aug.) |


### Demo

```
python demo_finger.py
```

## Isolated Sign Language Recognition

## Credits

Mathias Müller ([@bricksdont](https://github.com/bricksdont)) proposes the [initial idea](https://docs.google.com/document/d/1mUSLZs_DWc4mHn_nt0soKf1hsTtbrHUUnEX_QBCth5w/edit#heading=h.p699gptqhse9) of a CLIP model for sign language.
