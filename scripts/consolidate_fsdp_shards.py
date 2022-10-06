#!/usr/bin/env python
import fire

from fairseq.distributed.stitch_fsdp_ckpt import consolidate_fsdp_shards

if __name__ == "__main__":
    # This is expected to be used before evaluation, not during training.
    fire.Fire(consolidate_fsdp_shards)
