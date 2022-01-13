### Config Files Explained

Taking `projects/mfmmlm.yaml` for example, which run pretraining using masked frame model (MFM) and masked language model (MLM) on a single BERT:  

```yaml
project_dir: mfmmlm # specify the project dir for this baseline.
run_task:
  - how2.yaml # run pretraining on how2 when launching `projects/taskmfmmlm.yaml`
  - [vtt.yaml, vttcap.yaml, vttqa.yaml, youcook.yaml, youcookcap.yaml, crosstask.yaml, coin.yaml] # run fine-tuning tasks.
base_dir: task # a global template folder to specify each training task. 
task_group:
  pretrain: # section for pretraining. Most baselines differs in this section.
    task_list:
      - how2.yaml # reconfig `projects/task/how2.yaml`
    dataset:
      aligner: MFMMLMAligner # overwrite the aligner for MFMMLM training task.
    model:
      model_cls: MMFusionMFMMLM # overwrite the model, which constructs negative examples for MFM on-the-fly.
    loss:
      loss_cls: MFMMLM # overwrite the loss as MFMMLM, which combines MFM and MLM together.
    fairseq: # all fairseq args can be expecified under this name.
      dataset:
        batch_size: 128
  finetune: # section for fine-tuning tasks, we don't need to change anything here mostly since we want to see how pretraining can contribute to finetuning.
    task_list: # specify the list of downstream tasks, e.g., copy `projects/task/vtt.yaml` to `projects/mfmmlm`.
      - vtt.yaml
      - vttqa.yaml
      - youcook.yaml
      - youcookcap.yaml
      - crosstask.yaml
      - coin.yaml
  test: # section for testing.
    task_list:
      - test_vtt.yaml
      - test_vttqa.yaml
      - test_youcook.yaml
      - test_youcookcap.yaml
      - test_crosstask.yaml
      - test_crosstask_zs.yaml
      - test_coin.yaml
```
