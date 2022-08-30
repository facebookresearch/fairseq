# Noise and audio augmentation techniques

The noise and data augmentation techniques were written in an effort to understand how augmenatation can affect model robustness and performance in both clean and noisy settings. 

All transforms discussed in this section are subclasses of `AudioFeatureTransform`, `AudioWaveformTransform`, or `AudioDatasetTransform`. Each `Audio*Transform` has unique interaction with the data. If interested in implemented one's own transforms, it is highly advisable to review the differences [TODO: add link to section]. If only applying the in-built transforms, then one only needs to be mindful that the correct kind of transform is listed in the config (see [Using these transforms](https://github.com/fairinternal/fairseq-py/new/adishree_internship/examples#using-these-transforms) ). These transforms can be applied to instances of `SpeechToTextDataset`.

### Contents [TODO: add links]
[In-built transforms]()

[Benchmark studies]()

[Using transforms]()

[Adding your own transforms]()


## In-built transforms 
### 1. Utterance concatenation 
Utterance concatenation is a data augmenation technique introduced as ConcatAug in [Translatotron 2: High-quality direct speech-to-speech translation
with voice preservation](https://arxiv.org/pdf/2107.08661.pdf).
With some parameterized probability, samples are concatenated with one other randomly chosen sample from the whole dataset. In the positive (concatenation) case, accessing `dataset[i]` will return a `SpeechToTextDatasetItem` where `source=source[i]+source[j]` and `target=target[i]+target[j]`. In the negative (skip concatenation) case, accessing `dataset[i]` will return a `SpeechToTextDatasetItem` where `source=source[i]` and `target=target[i]` as usual. 

**Usage**: `concataugment` is an `AudioDatasetTransform` and has three configurable hyperparameters:
- `rate`: probability that any single access will result in the positive (concatenation) case. Defaults to 0.25. 
- `max_tokens`: maximum number of tokens allowed for concatenated source sequences. This parameter is meant to limit the length of concatenated samples to avoid out-of-memory errors. Defaults to 300. 
- `attempts`: maximum number of invalid concatenation attempts before defaulting to the negative (skip concatenation) case. This parameter aims to limit excessive time spent trying to find candidate samples that are short enough to concatenate with. Defaults to 5.

Please be wary of OOMs while using this augmentation technique; we used smaller batch sizes as a workaround to avoid OOMs. Batch size is determined by update frequency, batch size hyperparameter, and the number of GPU, so you may want to alter these to this end.

### 2. Noise augmentation suite 

The four noise augmentation methods in this suite adhere to the following principle: with some parameterized probability, samples are overlayed with a noise track. The content of the noise track is specific to the method. Signal-to-noise ratio with which the noise track is overlayed is determined by choosing a value from a random uniform distribution with parameterized endpoints. The first three methods are based off data augmentation methods suggested in Section 3.3 of [X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://danielpovey.com/files/2018_icassp_xvectors.pdf).

#### 2.1. Music augmentation
For music augmentation, the noise track consists of one file uniformly randomly selected from a corpus of music files. The music file is cut to size, including being repeated to fill the original sample length if necessary.  

**Usage**: `musicaugment` is an `AudioWaveformTransform` and has four configurable hyperparameters:
- `samples_path`: path where background music files are saved as audios (.wav files). No default. 
- `rate`: probability that any single access will result in the positive (background music) case. Defaults to 0.25. 
- `snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 5.
- `snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 15.

#### 2.2. Babble augmentation
For babble augmentation, the noise track consists of multiple audios uniformly randomly selected from a corpus of speech files. The number of speech audios in the background track is chosen randomly with equal probability between 3 and 7 audios.

**Usage**: `babbleaugment` is an `AudioWaveformTransform` and has four configurable hyperparameters:
- `samples_path`: path where background speech files are saved as audios (.wav files). No default. 
- `rate`: probability that any single access will result in the positive (background speech) case. Defaults to 0.25. 
- `snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 5.
- `snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 15.

#### 2.3. Sporadic noise augmentation
For sporadic noise augmentation, the noise track is mostly silent except for intermittent short clips of noise which are added at roughly a parameterized frequency. These clips are randomly chosen and cut from a corpus of noise files to lengths according to a parameterized Gaussian distribution.

**Usage**: `sporadicnoiseaugment` is an `AudioWaveformTransform` and has seven configurable hyperparameters:
- `samples_path`: path where background noise files are saved as audios (.wav files). No default.
- `rate`: probability that any single access will result in the positive (add a sporadic noise track) case. Defaults to 0.25.
- `snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 5.
- `snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 15.
- `noise_rate`: rate in noises per second at which noise clip will be added to the original sample
- `noise_len_mean`: mean of Gaussian normal distribution from which length of noise clip is chosen 
- `noise_len_std`: standard deviation of Gaussian normal distribution from which length of noise clip is chosen 

#### 2.4. Background noise augmentation
For background noise augmentation, the noise track is a single track uniformly randomly selected from a corpus of noise files. The noise file is cut to size, including being repeated to fill the original sample length if necessary.  

**Usage**: `backgroundnoiseaugment` is an `AudioWaveformTransform` and has four configurable hyperparameters:
- `samples_path`: path where background noise files are saved as audios (.wav files). No default. 
- `rate`: probability that any single access will result in the positive (background noise) case. Defaults to 0.25. 
- `snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 5.
- `snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 15.

### 3. Mixed babble and background noise augmentation with recognizable source speaker

This augmentation technique is based on Algorithm 1 in [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) and is similar to the noise augmentation suite techniques in that it has a background noise track. The noise track consists of either (1) another audio sample from the batch or (2) a background noise track. A key difference is the length of the noise track is chosen from a uniform random distribution between 0 and half of the original sample length. 

**Usage**: `noisyoverlapaugment` is an `AudioDatasetTransform` and has seven configurable hyperparameters:
- `noises_path`: path where background noise files are saved as audios (.wav files). No default. 
- `rate`: probability that any single access will result in the positive (background noise) case. Defaults to 0.25. 
- `mixing_noise_rate`: probability that in a positive (background noise) case, the noise track will consist of background noise (rather than babble from the batch). Defaults to 0.1.
- `noise_snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to -5.
- `noise_snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add background noise to the original source. Defaults to 5.
- `utterance_snr_min`: lower endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add **another audio from the batch** to the original source. Defaults to -5.
- `utterance_snr_max`: higher endpoint of the range from which a signal-to-noise ratio is uniformly randomly chosen with which to add **another audio from the batch** to the original source. Defaults to 5.

## Benchmark studies [TODO] 
### Clean data
| Augmentation | Hyperparameters | Loss | BLEU (covost) | BLEU (epst) | BLEU (mtedx) | Model | 
| --- | --- | --- | --- | --- | --- | --- |
| None | | | | | | |
| ConcatAugment | rate = 0.25, max_tokens = 3000, attempts = 5 | | | | | |
| MusicAugment | | | | | | |
| BabbleAugment | | | | | | |
| BackgroundNoiseAugment | | | | | | |
| SporadicNoiseAugment | | | | | | |
| MusicAugment + BabbleAugment + BackgroundNoiseAugment + SporadicNoiseAugment | | | | | | |
| NoisyOverlapAugment | | | | | | |

## Using transforms 
Transforms are configurable. 

1. Please pay careful attention to the type of transform you are applying. 
    - `concataugment` and `noisyoverlapaugment` are instances of `AudioDatasetTransform` and should be listed in the config under `dataset_transforms`.
    - `musicaugment`, `babbleaugment`, `sporadicnoiseaugment`, and `backgroundnoiseaugment` are instances of `AudioWaveformTransform` and should be listed under `waveform_transforms`.
    - Instances of `AudioFeatureTransform` should be listed under `feature_transforms`. 
2. Feel free to apply these augmentations in different contexts, e.g., you may use a `_train` or `_eval` flag to specify when the transform will be applied. If the dataset at hand contains `train` in its name, those transforms under the `_train` flag will be applied; else, the remaining transforms will be applied. 

For example, you would add this to your config to apply the musicaugment transform to a training dataset: 
```yaml
musicaugment:
  samples_path: ${MUSIC_PATH}
  snr_min: 10 
  snr_max: 15
  rate: 0.25
waveform_transforms:
  _train:
  - musicaugment
```
or add this to apply the concataugment transform: 
```yaml
concataugment:
  rate: 0.25
  max_tokens: 3000
  attempts: 5
dataset_transforms:
  _train:
  - concataugment
 ```
You may also want to add multiple of one type of transform; here, we add multiple `AudioWaveformTransform`s: 
```yaml
musicaugment:
  samples_path: ${MUSIC_PATH}
  snr_min: 5 
  snr_max: 20
  rate: 0.25
backgroundnoiseaugment:
  samples_path: ${NOISES_PATH}
  snr_min: 10
  snr_max: 20
  rate: 0.1
sporadicnoiseaugment:
  samples_path: ${NOISES_PATH}
  snr_min: 5
  snr_max: 15
  rate: 0.1
  noise_rate: 0.25
waveform_transforms:
  _train:
  - musicaugment
  - backgroundnoiseaugment
  - sporadicnoiseaugment
```

## Adding your own transforms
Note: We store transform implementations in `fairseq/data/audio/*_transforms` directories. You may refer to these as examples while implementing your own transform.

### Step 1. Picking the right class for your transform
The integration into SpeechToTextDataset is quite different for each kind of transform, so it is important to understand which one is best suited to your purposes. 

**Feature transforms**
`AudioFeatureTransform` is a base class which allows **some transform to be applied to audio spectrograms** in the data loading step. One thing to note is that the source data is either saved as `np.ndarrays` or as audio files, and is to be returned either as features (spectrogram) or waveform. If and only if the data is to be returned as a spectrogram, then `AudioFeatureTransform`s will be applied.

**Waveform transforms**
`AudioWaveformTransform` is a base class which allows some **transform to be applied to waveforms** in the data loading step. As mentioned above, there are two source and return types to data loading for this dataset. If and only if the data is saved in audio file format, then `AudioWaveformTransform`s will be applied, whichever return type is used.

**Dataset transforms**
`AudioDatasetTransform` is a base class for transforms **based on more than one item in a dataset**, ex. concatenation of two random samples in a dataset. Rather than being applied in a consistent way, i.e., to all features or to all waveforms, the integration of a dataset transform is entirely specific. Adding a dataset transform requires actually editing the `fairseq/data/audio/speech_to_text_dataset.py` file.

### Step 2. Setting up your transform (generic to all types of transforms)
Now that you know which kind of transform you would like to use, we are ready to implement it. This step is generic for all transform types, i.e., `TRANSFORM_TYPE` may be any of `feature`, `waveform`, or `dataset`. We will show how to build utterance concatenation (an `AudioDatasetTransform`) as an example. 

Import the base class and registration function for your transform. 
```python
from fairseq.data.audio.dataset_transforms import (
  AudioDatasetTransform,
  register_audio_dataset_transform
)
```

Define the class and register the transform. The name passed into the registration function is how your transform should be named in the config.
```python
@register_audio_dataset_transform("concataugment")
class ConcatAugment(AudioDatasetTransform):
```

We are now ready to add the basic important functions to our new class. In this example, `_DEFAULTS` refers to a dictionary with the default hyperparameter values that we defined. `from_config_dict` is called to instantiate the transform given hyperparameters from the config. 
```python
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        return ConcatAugment(
            _config.get("rate", _DEFAULTS["rate"]),
            _config.get("max_tokens", _DEFAULTS["max_tokens"]),
            _config.get("attempts", _DEFAULTS["attempts"]),
        )
```
We edit the instantiation function `__init__` to track hyperparameters and do any setup work.
```python
    def __init__(
        self,
        rate=_DEFAULTS["rate"],
        max_tokens=_DEFAULTS["max_tokens"],
        attempts=_DEFAULTS["attempts"],
    ):
        self.rate, self.max_tokens, self.attempts = rate, max_tokens, attempts
```
Lastly `__repr__` gives how the transform will be reported in an output log. 
```python
    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                [
                    f"rate={self.rate}",
                    f"max_tokens={self.max_tokens}",
                    f"attempts={self.attempts}",
                ]
            )
            + ")"
        )
```

### Step 3. Adding the transform logic 
At this point, we are ready to implement the actual transform logic. The flow from here is different for each of the three transforms, so follow the path that is relevant to you.
### ...for feature transforms
The final step is implementing the `__call__` function, which applies the transform logic and **returns** the spectrogram with transform applied. This supports and should take exactly **two arguments**: 
- `self`
- `x` (np.ndarray): the spectrogram for one source sample. (This is a positional argument, so you can use another parameter name like `spectrogram` instead of `x`.)

For example, this is the `__call__` function for GlobalCMVN (cepstral mean and variance normalization). 
```python
    def __call__(self, x):
        x = np.subtract(x, self.mean)
        x = np.divide(x, self.std)
        return x

```
### ...for waveform transforms
The final step is implementing the `__call__` function, which applies the transform logic. This supports and should take exactly **three arguments**: 
- `self`
- `source` (numpy.ndarray or torch.Tensor): source audio 2d waveform (channels x length)
- `sample_rate` (optional, defaults to None): sample rate of `source`

`__call__` **returns**:
- transformed audio waveform 
- sample rate of transformed audio waveform

For example, this is the `__call__` function for augmentations in the Noise Augmentation Suite. 
```python
    def __call__(self, source, sample_rate=None):
        if np.random.random() > self.rate:
            return source

        noise = self._get_noise(
            source.shape, always_2d=True, use_sample_rate=sample_rate
        )
        return self._mix(source, noise, rand_uniform(self.snr_min, self.snr_max)), sample_rate
```

### ...for dataset transforms
Dataset transforms are extremely flexible, and implementation involves directly integrating them into `fairseq/data/audio/speech_to_text_dataset.py` in transform-specific ways. 
There are two basic components: (1) check whether or not this transform is part of this dataset instance using `self.dataset_transforms.has_transform(TRANSFORM_CLS)`, and (2) if so, get the transform using `self.dataset_transforms.get_transform(TRANSFORM_CLS)` & apply it.
Due to the case-by-case specificity, it is easier to demonstrate this by examples. 

#### Example: NoisyOverlapAugment 
This transform requires access to multiple items within the same batch at once. 

**Logic**: We still use the transform classes to keep away the transform logic. For example, `__call__` of `NoisyOverlapAugment` class takes a list of source tokens for items in a mini-batch, applies noise/utterance as dictated by the transform, and returns the list of transformed source tokens for items in the mini-batch.

```python
    def __call__(self, sources):
        for i, source in enumerate(sources):
            if np.random.random() > self.rate:
                continue

            pri = source.numpy()

            # ... some transform code omitted 
            
            pri[s_source : s_source + l] = np.add(
                pri[s_source : s_source + l], np.multiply(scl, sec[s_sec : s_sec + l])
            )
            sources[i] = torch.from_numpy(pri).float()

        return sources
```

**Integration**: The `collater` function for `SpeechToTextDataset` is responsible for preparing a mini-batch for training, so we integrate NOAug through adding a few lines to the top of this function: 
```python
def collater(
    self, samples: List[SpeechToTextDatasetItem], return_order: bool = False
) -> Dict:
    if len(samples) == 0:
        return {}
    indices = torch.tensor([x.index for x in samples], dtype=torch.long)

    sources = [x.source for x in samples]

    # NOAUG INTEGRATION BLOCK
    # (1) Check whether or not this transform is part of this dataset instance
    has_NOAug = self.dataset_transforms.has_transform(NoisyOverlapAugment)
    # (2) If so, get & apply the transform
    if has_NOAug and self.cfg.use_audio_input:
        NOAug = self.dataset_transforms.get_transform(NoisyOverlapAugment)
        sources = NOAug(sources)

    frames = _collate_frames(sources, self.cfg.use_audio_input)
    # sort samples by descending number of frames
    n_frames = torch.tensor([x.size(0) for x in sources], dtype=torch.long)
    n_frames, order = n_frames.sort(descending=True)
    indices = indices.index_select(0, order)
    frames = frames.index_select(0, order)

    # ... rest of function
```

#### Example: ConcatAugment
This transform requires access to another item within the dataset at once. 

**Logic**: We abstract the logic for picking indices to concatenate by adding a `find_indices` function to the `ConcatAugment` class, which takes one index in the dataset and finds a compatible second index to concatenate source and target tokens.
```python
    def find_indices(self, index: int, n_frames: List[int], n_samples: int):
        # skip conditions: application rate, max_tokens limit exceeded
        if np.random.random() > self.rate:
            return [index]
        if self.max_tokens and n_frames[index] > self.max_tokens:
            return [index]

        # pick second sample to concatenate
        for _ in range(self.attempts):
            index2 = np.random.randint(0, n_samples)
            if index2 != index and (
                not self.max_tokens
                or n_frames[index] + n_frames[index2] < self.max_tokens
            ):
                return [index, index2]

        return [index]
```

**Integration**: `SpeechToTextDataset` uses a custom `__getitem__(self, index)` function (called in the background when you write `dataset[i]`). We edited this function (as well as `_get_source_audio` and `get_tokenized_tgt_text`) to achieve the desired transform effect where accessing `dataset[i]` will return a `SpeechToTextDatasetItem` where `source=source[i]+source[j]` and `target=target[i]+target[j]`.
```python
def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
    
    # CONCATAUGMENT INTEGRATION BLOCK
    # (1) Check whether or not this transform is part of this dataset instance
    has_concat = self.dataset_transforms.has_transform(ConcatAugment)
    # (2) If so, get & apply the transform
    if has_concat:
        concat = self.dataset_transforms.get_transform(ConcatAugment)
        indices = concat.find_indices(index, self.n_frames, self.n_samples)

    source = self._get_source_audio(indices if has_concat else index)
    source = self.pack_frames(source)

    target = None
    if self.tgt_texts is not None:
        tokenized = self.get_tokenized_tgt_text(indices if has_concat else index)
        target = self.tgt_dict.encode_line(

    # ... rest of function
```
