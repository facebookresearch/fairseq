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
print(data2vec.keys())
# dict_keys(['args', 'cfg', 'model', 'criterion', 'optimizer_history', 'task_state', 'extra_state', 'last_optimizer_state'])
print(data2vec['model'].keys())
print(
    data2vec['model']['_ema'].keys()
)  # we can see that in ema, there are only encoder. because self.cfg.ema_transformer_only is True
print(data2vec['cfg'].keys())
print(data2vec['cfg']['ema'])

# for item in data2vec['model'].keys():
#     print(item, data2vec['model'][item].shape) # student model

# for item in data2vec['model']['_ema'].keys():
#     print(item, data2vec['model']['_ema'][item].shape) # teacher model
cfg = Data2VecAudioConfig(**data2vec['cfg'])
teacher_dict = data2vec['model']['_ema']  # ema teacher model
teacher_model = TransformerEncoder(cfg)
teacher_model.load_state_dict(teacher_dict, strict=True)
teacher_cfg = EMAModuleConfig(**data2vec['cfg']['ema'])
print(teacher_cfg)
teacher_model = EMAModule(teacher_model, teacher_cfg)
print(teacher_model)

# student_dict = data2vec['model'].pop('_ema') # pure student model
