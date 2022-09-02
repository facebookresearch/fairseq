# Zero-shot Transfer and Finetuning

(If you are new to the ideas of `mmpt.processors`, see [README](README.md) first.)
All finetuning datasets (specifically `processors`) are defined in `mmpt.processors.dsprocessor`.
Given the complexity of different types of finetuning tasks, each task may have their own meta/video/text/aligner processors and `mmpt/evaluators/{Predictor,Metric}`.

### Tasks

Currently, we support 5 end datasets: `MSRVTT`, `Youcook`, `COIN`, `Crosstask` and `DiDeMo` with the following tasks:  
text-video retrieval: `MSRVTT`, `Youcook`, `DiDeMo`;   
video captioning: `Youcook`;  
Video Question and Answering: `MSRVTT-QA`.  

To add your own dataset, you can specify the corresponding processors and config them in the `dataset` field of a config file, such as `projects/task/vtt.yaml`.

### Zero-shot Transfer (no Training)
Zero-shot transfer will run the pre-trained model (e.g., VideoCLIP) directly on testing data. Configs with pattern: `projects/task/*_zs_*.yaml` are dedicated for zero-shot transfer.

### Fine-tuning

The training of a downstream task is similar to pretraining, execept you may need to specify the `restore_file` in `fairseq.checkpoint` and reset optimizers, see `projects/task/ft.yaml` that is included by `projects/task/vtt.yaml`.

We typically do finetuning on 2 gpus (`local_small`).

### Testing
For each finetuning dataset, you may need to specify a testing config, similar to `projects/task/test_vtt.yaml`.  

We define `mmpt.evaluators.Predictor` for different types of prediction. For example, `MSRVTT` and `Youcook` are video-retrieval tasks and expecting to use `RetrievalPredictor`. You may need to define your new type of predictors and specify that in `predictor` field of a testing config.

Each task may also have their own metric for evaluation. This can be created in `mmpt.evaluators.Metric` and specified in the `metric` field of a testing config.

Launching a testing is as simple as training by specifying the path of a testing config:
```python locallaunch.py projects/mfmmlm/test_vtt.yaml```
Testing will be launched locally by default since prediction is computationally less expensive.

### Third-party Libraries
We list the following finetuning tasks that require third-party libraries.

Youcook captioning: `https://github.com/Maluuba/nlg-eval`  

CrossTask: `https://github.com/DmZhukov/CrossTask`'s `dp` under `third-party/CrossTask` (`python setup.py build_ext --inplace`)
