from torchaudio.compliance.kaldi import fbank
import soundfile as sf
import torch
import numpy as np
from fairseq.modules import conformer_layer
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
from fairseq.modules.conformer_layer import ConformerEncoderLayer
import wave

from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel, ConvFeatureExtractionModel2D

#from ..models.conformer import Conformer

input_data, sample_rate = sf.read(
    '/mnt/c/Users/86181/Desktop/3607-135982-0000.flac')
with torch.no_grad():
    x = torch.from_numpy(input_data).float()
    x = x.view(1, -1)

print(x.shape)  # raw data (1, 108160)
print(sample_rate)  # sample rate

mask_x = torch.clone()

mask_start_points = torch.randint(0, x.shape[1],
                                  (x.shape[1] / 100, )).sort().values
mask_points = sample_rate * 0.4  # mask points(400ms)
for point in mask_start_points:
    mask_x[point:point + mask_points] = torch.normal(
        mean=0, std=0.1,
        size=(mask_points, ))  # use normal distribution to mask
# to mask the data, length 400ms with probablity 0.01

fbank_feat = fbank(waveform=mask_x, frame_shift=10, num_mel_bins=80)
print(fbank_feat.shape)  # fbank (674, 80)
fbank_feat = torch.cat([fbank_feat, fbank_feat])  # batched fbank (2,674,80)

# specaug
specaug_Transform = SpecAugmentTransform() # TODO: add params
specaug_feat = specaug_Transform(fbank_feat)

# conv_subsampling
conv_subsampling = ConvFeatureExtractionModel(conv_layers=[(512, 2, 2), (512, 2, 2)])
features = conv_subsampling(specaug_feat)
print(features.shape)

# dropout & linear
linear_1 = torch.nn.Linear(in_features=80, out_features=1024)
features = linear_1(features)

# linear: random_projection
linear1 = torch.nn.Linear(80, )
random_projection = torch.randn([80, 512])

# conformer
conformer = torch.nn.ModuleList(
            [ConformerEncoderLayer(
                embed_dim=1024,
                ffn_embed_dim=1024,
                attention_heads=12,
                dropout=0.1,
            ) for _ in range(24)]
        )


#conformer = Conformer(input_dim=cfg.encoder_embed_dim,num_heads=cfg.encoder_attention_heads,ffn_dim=cfg.encoder_ffn_embed_dim,num_layers=cfg.encoder_layers,depthwise_conv_kernel_size=cfg.depthwise_conv_kernel_size,dropout=cfg.dropout_features)