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

## Credits

Mathias Müller ([@bricksdont](https://github.com/bricksdont)) proposes the [initial idea](https://docs.google.com/document/d/1mUSLZs_DWc4mHn_nt0soKf1hsTtbrHUUnEX_QBCth5w/edit#heading=h.p699gptqhse9).
