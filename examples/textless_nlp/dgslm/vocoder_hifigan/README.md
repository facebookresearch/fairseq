# Dialogue Unit-to-Speech Decoder for dGSLM
For the unit2speech decoder, we train a [discrete unit-based HiFi-GAN vocoder](https://arxiv.org/pdf/2104.00355.pdf) on the [Fisher dataset](http://www.lrec-conf.org/proceedings/lrec2004/pdf/767.pdf).

The pre-trained model checkpoint can be found here :

| HiFi-GAN vocoder based on HuBERT Fisher Units |
|-----------------------------------------------|
|[model checkpoint](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hifigan/hifigan_vocoder) - [config](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hifigan/config.json) |

To create waveform from discrete units, use the script `generate_stereo_waveform.py` :
```bash
python examples/textless_nlp/dgslm/vocoder_hifigan/generate_stereo_waveform.py \
    --in-file $INPUT_CODE_FILE \
    --vocoder $VOCODER_PATH \
    --vocoder-cfg $VOCODER_CONFIG \
    --results-path $OUTPUT_DIR
```
where INPUT_CODE_FILE is expected to have the following format :
```
{'audio': 'file_1', 'unitA': '8 8 ... 352 352', 'unitB': '217 8 ... 8 8'}
{'audio': 'file_2', 'unitA': '5 5 ... 65 65', 'unitB': '6 35 ... 8 9'}
...
```