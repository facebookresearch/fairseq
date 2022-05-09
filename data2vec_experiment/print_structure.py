import torch

from fairseq.modules.ema_module import EMAModule, EMAModuleConfig
from fairseq.examples.data2vec.models.data2vec_audio import Data2VecAudioModel,Data2VecAudioConfig
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)

model_path = "/mnt/c/Users/86181/Desktop/checkpoint_423_400000.pt"
data2vec = torch.load(model_path)
print('all keys:')
print(data2vec.keys())
# dict_keys(['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state'])
print('student keys:')
print(data2vec['model'].keys())
print('teacher keys:')
print(
    data2vec['model']['_ema'].keys()
)  # we can see that in ema, there are only encoder. because self.cfg.ema_transformer_only is True