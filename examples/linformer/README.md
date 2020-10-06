# Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)

This example contains code to train Linformer models as described in our paper
[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768).

## Training a new Linformer RoBERTa model

You can mostly follow the [RoBERTa pretraining README](/examples/roberta/README.pretraining.md),
but replace the architecture with `--arch linformer_roberta_base` in your training command.

## Citation

If you use our work, please cite:

```bibtex
@article{wang2020linformer,
  title={Linformer: Self-Attention with Linear Complexity},
  author={Wang, Sinong and Li, Belinda and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
```
