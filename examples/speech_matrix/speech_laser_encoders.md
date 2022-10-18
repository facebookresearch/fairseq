# SpeechMatrix: A Large-Scale Mined Corpus of Multilingual Speech-to-Speech Translations


## SpeechLASER encoders

To embed an audio into LASER sentence embedding space with speechLASER encoders used for SpeechMatrix project:

```bash
from fairseq.models.wav2vec import Wav2VecLaser
import torch.nn.functional as F
import torch
import soundfile as sf

model = Wav2VecLaser.from_pretrained(CHECKPOINT_DIR, checkpoint_file=CHECKPOINT_NAME).models[0]
model.eval()
wav, sr = sf.read(PATH_TO_WAV_FILE) # sr needs to be 16000
feats = torch.from_numpy(wav).float()

with torch.no_grad():
    feats = F.layer_norm(feats, feats.shape).unsqueeze(0)
    padding_mask = torch.Tensor([False]*feats.shape[1])
    sample = {'padding_mask': padding_mask, 'source': feats}
    embedding = model(**sample)
```

Please download the speechLASER encoders trained for the different language families:

Languages | url
|---|---|
**Romance** - ca, es, fr, it, pt, ro | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/romance.pt) 
**Slavic** - ru, cs, pl, sk, hr, (sl) | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/slavic.pt)
**Germanic** - de, (nl, en) | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/germanic.pt)
**Uralic** - fi, et, hu | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/uralic.pt)
**English** | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/english.pt) 
**Slovenian** | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/slovenian.pt) 
**Lithuanian** | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/lithuanian.pt) 
**Dutch** | [ckpt](https://dl.fbaipublicfiles.com/speechlaser_encoders/dutch.pt) 



