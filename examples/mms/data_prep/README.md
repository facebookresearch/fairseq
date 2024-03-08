# Data Preparation 

We describe the process of aligning long audio files with their transcripts and generating shorter audio segments below. 

- Step 1:  Download and install torchaudio using the nightly version. We have open sourced the CTC forced alignment algorithm described in our paper via [torchaudio](https://github.com/pytorch/audio/pull/3348). 
  ```
  pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
  ```
  
- Step 2: Download [uroman](https://github.com/isi-nlp/uroman) from Github. It is a universal romanizer which converts text in any script to the Latin alphabet. Use [this link](https://www.isi.edu/~ulf/uroman.html) to try their web interface.  
  ```
  git clone git@github.com:isi-nlp/uroman.git
  ```
  
- Step 3: Install a few other dependencies 
  ```
  apt install sox 
  pip install sox dataclasses 
  ```  

- Step 4: Create a text file containing the transcript for a (long) audio file. Each line in the text file will correspond to a separate audio segment that will be generated upon alignment.

  Example content of the input text file :
  ```
  Text of the desired first segment
  Text of the desired second segment
  Text of the desired third segment
  ```

- Step 5: Run forced alignment and segment the audio file into shorter segments. 
  ```
  python align_and_segment.py --audio /path/to/audio.wav --text_filepath /path/to/textfile --lang <iso> --outdir /path/to/output --uroman /path/to/uroman/bin 
  ```

  The above code  will generated the audio segments under output directory based on the content of each line in the input text file. The `manifest.json` file consisting of the of segmented audio filepaths and their corresponding transcripts. 

  ```
  > head /path/to/output/manifest.json 

  {"audio_start_sec": 0.0, "audio_filepath": "/path/to/output/segment1.flac", "duration": 6.8, "text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "normalized_text": "she wondered afterwards how she could have spoken with that hard serenity how she could have", "uroman_tokens": "s h e w o n d e r e d a f t e r w a r d s h o w s h e c o u l d h a v e s p o k e n w i t h t h a t h a r d s e r e n i t y h o w s h e c o u l d h a v e"}
  {"audio_start_sec": 6.8, "audio_filepath": "/path/to/output/segment2.flac", "duration": 5.3, "text": "gone steadily on with story after story poem after poem till", "normalized_text": "gone steadily on with story after story poem after poem till", "uroman_tokens": "g o n e s t e a d i l y o n w i t h s t o r y a f t e r s t o r y p o e m a f t e r p o e m t i l l"}
  {"audio_start_sec": 12.1, "audio_filepath": "/path/to/output/segment3.flac", "duration": 5.9, "text": "allan's grip on her hands relaxed and he fell into a heavy tired sleep", "normalized_text": "allan's grip on her hands relaxed and he fell into a heavy tired sleep", "uroman_tokens": "a l l a n ' s g r i p o n h e r h a n d s r e l a x e d a n d h e f e l l i n t o a h e a v y t i r e d s l e e p"}
  ```

  To visualize the segmented audio files, [Speech Data Explorer](https://github.com/NVIDIA/NeMo/tree/main/tools/speech_data_explorer) tool from NeMo toolkit can be used.  

  As our alignment model outputs uroman tokens for input audio in any language, it also works with non-english audio and their corresponding transcripts. 
