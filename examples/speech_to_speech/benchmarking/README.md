# Benchmarking

## Overview

The goal of this framework is to support benchmarking various speech to speech translation(S2ST) models in terms of runtime, max-memory consumption and total number of floating point operations(FLOPS). It is a generic framework and can be easily extended to support any fairseq models. To accurately benchmark the performance, core inference modules are re-implemented based on fairseq_cli/generate.py (core.py/Processing) and examples/speech_to_text/generate_waveform.py(core.py/SpeechGeneration. To ensure that the end to end models and cascaded models are compared fairly, for cascaded models we only consider the performance metrics for model inference at all stages ignoring any intermediate data and io processing consumption. We run all the benchmarking runs on CPU as it is generally used in production environment and also due to lack of good benchmarking library support for GPUs.

1. Runtime: Average time in seconds to run model inference on an example from a given dataset. We use [timeit](https://docs.python.org/3/library/timeit.html) library to measure the runtime.
2. Max memory: Maximum memory in MiB averaged over by running the model inference on all examples from the given dataset. We use [memory_profiler](https://pypi.org/project/memory-profiler/) library to gather memory footprints for a code snippet and find the maximum to get the max memory used by the code. For cascaded models, we find the max of all stages to get the overall max_memory footprint.
3. FLOPS: We compute the average number of floating point operations needed to run model inference for an example from the given dataset. We use [PAPI library](http://www.bnikolic.co.uk/blog/python/flops/2019/10/01/pytorch-count-flops.html) to benchmark the number of flops.

## CLI Commands

```{python}
CUBLAS_WORKSPACE_CONFIG=:4096:8 python examples/speech_to_speech/benchmarking/get_metrics.py ‘’ --config $config
```


## Note:

1. The npy dataset is a list of samples saved as a .npy file. Each sample is a dictionary with id, net_input.
2. The raw dataset is a list of raw audio paths similar to wav2vec2 input tsv file

```{python}
sample: {
    "id": xx,
    "net_input": {
        "src_tokens": torch.tensor([]),
        "src_lengths": torch.tensor([])
    }
}
```
