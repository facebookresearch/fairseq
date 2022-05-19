# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from .. import tasks
from .. import models
from .. import losses
from ..datasets import MMDataset
from .. import processors


class Task(object):
    """
    A task refers to one generic training task (e.g., training one model).
    """

    @classmethod
    def config_task(cls, config):
        """
        determine whether to load a hard-coded task or config from a generic one.
        via if a task string is available in config.
        """
        if config.task is not None:
            # TODO (huxu): expand the search scope.
            task_cls = getattr(tasks, config.task)
            return task_cls(config)
        else:
            return Task(config)

    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.model = None
        self.loss_fn = None
        self.eval_fn = None

    def build_dataset(self):
        """TODO (huxu): move processor breakdown to MMDataset."""
        """fill-in `self.train_data`, `self.val_data` and `self.test_data`."""

        meta_processor_cls = getattr(
            processors, self.config.dataset.meta_processor)
        video_processor_cls = getattr(
            processors, self.config.dataset.video_processor)
        text_processor_cls = getattr(
            processors, self.config.dataset.text_processor)
        aligner_cls = getattr(
            processors, self.config.dataset.aligner)

        if self.config.dataset.train_path is not None:
            self.config.dataset.split = "train"
            # may be used by meta processor.
            # meta_processor controls different dataset.
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)
            aligner = aligner_cls(self.config.dataset)
            self.train_data = MMDataset(
                meta_processor, video_processor, text_processor, aligner
            )
            print("train_len", len(self.train_data))
            output = self.train_data[0]
            self.train_data.print_example(output)
        if self.config.dataset.val_path is not None:
            self.config.dataset.split = "valid"
            # may be used by meta processor.
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)
            aligner = aligner_cls(self.config.dataset)
            self.val_data = MMDataset(
                meta_processor, video_processor, text_processor, aligner
            )
            print("val_len", len(self.val_data))
            output = self.val_data[0]
            self.val_data.print_example(output)

        if self.config.dataset.split == "test":
            # the following is run via lauching fairseq-validate.
            meta_processor = meta_processor_cls(self.config.dataset)
            video_processor = video_processor_cls(self.config.dataset)
            text_processor = text_processor_cls(self.config.dataset)

            self.test_data = MMDataset(
                meta_processor, video_processor, text_processor, aligner
            )
            print("test_len", len(self.test_data))
            output = self.test_data[0]
            self.test_data.print_example(output)

    def build_model(self, checkpoint=None):
        if self.model is None:
            model_cls = getattr(models, self.config.model.model_cls)
            self.model = model_cls(self.config)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
        return self.model

    def load_checkpoint(self, checkpoint):
        if self.model is None:
            raise ValueError("model is not initialized.")
        state_dict = torch.load(checkpoint)
        state_dict = self._trim_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        # if it's a fp16 model, turn it back.
        if next(self.model.parameters()).dtype == torch.float16:
            self.model = self.model.float()
        return self.model

    def _trim_state_dict(self, state_dict):
        from collections import OrderedDict

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "model" in state_dict:  # fairseq checkpoint format.
            state_dict = state_dict["model"]
        ret_state_dict = OrderedDict()
        for (
            key,
            value,
        ) in state_dict.items():
            # remove fairseq wrapper since this is a task.
            if key.startswith("mmmodel"):
                key = key[len("mmmodel."):]
            ret_state_dict[key] = value
        return ret_state_dict

    def build_loss(self):
        if self.loss_fn is None and self.config.loss is not None:
            loss_cls = getattr(losses, self.config.loss.loss_cls)
            self.loss_fn = loss_cls()
        return self.loss_fn

    def flat_subsample(self, tensor):
        size = tensor.size()
        if len(size) >= 2:
            batch_size = size[0] * size[1]
            expanded_size = (
                (batch_size,) + size[2:] if len(size) > 2
                else (batch_size,)
            )
            tensor = tensor.view(expanded_size)
        return tensor

    def reshape_subsample(self, sample):
        if (
            hasattr(self.config.dataset, "subsampling")
            and self.config.dataset.subsampling is not None
            and self.config.dataset.subsampling > 1
        ):
            for key in sample:
                if torch.is_tensor(sample[key]):
                    sample[key] = self.flat_subsample(sample[key])
        return sample

    def __call__(self, model, sample):
        loss = None
        loss_scalar = float("inf")

        sample = self.reshape_subsample(sample)
        outputs = self.model(**sample)
        sample.update(outputs)
        if self.loss_fn is not None:
            loss = self.loss_fn(**sample)
            loss_scalar = loss.item()

        batch_size = sample["caps"].size(0)
        sample_size = 1
        return {
            "loss": loss,
            "loss_scalar": loss_scalar,
            "max_len": self.config.dataset.max_len,
            "batch_size": batch_size,
            "sample_size": sample_size,
        }

    def build_dataloader(self):
        """only used for trainer that lacks building loaders."""
        raise NotImplementedError
