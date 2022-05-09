# to generate a teacher(by old student) model
import torch

from fairseq.modules.ema_module import EMAModule, EMAModuleConfig
from fairseq.examples.data2vec.models.data2vec_audio import Data2VecAudioModel,Data2VecAudioConfig
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)

old_model_path = "/mnt/c/Users/86181/Desktop/checkpoint_423_400000.pt"
old_data2vec = torch.load(old_model_path)
new_model_path = "/mnt/c/Users/86181/Desktop/checkpoint_423_400000.pt"
new_data2vec = torch.load(new_model_path)

teacher_dict = old_data2vec['model']
teacher_dict_modify = dict()
for k,v in teacher_dict.items():
    if('encoder.' in k):
        teacher_dict_modify[k[8:]] = v

print(teacher_dict_modify.keys())
print(teacher_dict['_ema'].keys())

print(teacher_dict_modify.keys() == teacher_dict['_ema'].keys() )

new_data2vec['model']['_ema'] = teacher_dict_modify # set teacher model

mixed_model_path = "/mnt/c/Users/86181/Desktop/mixed.pt"
torch.save(new_data2vec, mixed_model_path)
