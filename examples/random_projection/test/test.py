import soundfile as sf
import torch
import numpy as np
from torch import layer_norm
from fairseq.modules import conformer_layer
from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
from fairseq.modules.conformer_layer import ConformerEncoderLayer
import wave

from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel

#from ..models.conformer import Conformer


# input data
input_data, _ = sf.read(
    '/home/nullptr/open-source/fairseq/examples/random_projection/test/3607-135982-0000.flac'
)
input_data = torch.Tensor(input_data).unsqueeze(0)
print('raw input data:', input_data.shape)

x = torch.cat([input_data,input_data,input_data],dim=0)
print('batch input data:',x.shape) # [3, 108160]

# feature extract

feature_enc_layers = eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]")
extractor_embed = feature_enc_layers[-1][0]

feature_extractor = ConvFeatureExtractionModel(
    conv_layers=feature_enc_layers,
    dropout=0.0,
    mode="default",
    conv_bias=False,
)

features = feature_extractor(x)
print('feature data:', features.shape) # [3, 512, 337]

features = features.transpose(1, 2)
layer_norm = torch.nn.LayerNorm(extractor_embed)
features = layer_norm(features)

unmasked_features = features.clone()

post_proj = torch.nn.Linear(extractor_embed, 768)
x = post_proj(features)

# conformer
conformer = torch.nn.ModuleList([
    ConformerEncoderLayer(
        embed_dim=768,
        ffn_embed_dim=768,
        attention_heads=12,
        dropout=0.0,
        use_fp16=False,
    ) for _ in range(24)
])
layer_results = []
position_emb = None
padding_mask = None
r = None
for i, layer in enumerate(conformer):
    x, z = layer(
                    x,
                    encoder_padding_mask=padding_mask,
                    position_emb=position_emb,
                )
    #if tgt_layer is not None:
    layer_results.append((x, z))
    if i == 23:
        r = x
        break
print('conformer data:',r.shape)

final_proj = torch.nn.Linear(768, 8192)
final = final_proj(r)
print('final proj data:',final.shape)
final = final.transpose(1, 2)
print('final proj data:',final.shape)

# random projection
random_proj = torch.nn.Parameter(torch.empty(512, 16), requires_grad=False) # embedding_dim, codebook_dim
codebook = torch.nn.Parameter(torch.empty(8192, 16), requires_grad=False)
random_proj = torch.nn.init.xavier_normal_(random_proj)
codebook = torch.nn.init.xavier_normal_(codebook)
print('random projection:', random_proj.shape)
print('codebook:', codebook.shape)

with torch.no_grad():
    proj_vector = torch.matmul(unmasked_features,random_proj)
    print('proj vector:', proj_vector.shape)
    # mask_num = proj_vector.shape[1]
    # random_codebook = codebook.unsqueeze(1).repeat(1,mask_num,1,1)
    # print('random codebook:', random_codebook.shape)
    proj_vector = torch.unsqueeze(proj_vector, 2)
    print('proj vector:', proj_vector.shape)
    norm2 = torch.squeeze(torch.sum(torch.sub(codebook, proj_vector) ** 2, dim=-1))
    print('norm2:', norm2.shape)
    labels = torch.argmin(norm2, dim=-1)
    print('labels:', labels.shape)
    # print(labels)

criterion = torch.nn.CrossEntropyLoss()
loss = criterion(final,labels)
print(loss)