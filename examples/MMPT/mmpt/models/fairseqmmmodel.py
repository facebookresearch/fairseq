# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture
)


@register_model("mmmodel")
class FairseqMMModel(BaseFairseqModel):
    """a fairseq wrapper of model built by `task`."""

    @classmethod
    def build_model(cls, args, task):
        return FairseqMMModel(task.mmtask.model)

    def __init__(self, mmmodel):
        super().__init__()
        self.mmmodel = mmmodel

    def forward(self, *args, **kwargs):
        return self.mmmodel(*args, **kwargs)

    def upgrade_state_dict_named(self, state_dict, name):

        super().upgrade_state_dict_named(state_dict, name)

        keys_to_delete = []

        for key in state_dict:
            if key not in self.state_dict():
                keys_to_delete.append(key)
        for key in keys_to_delete:
            print("[INFO]", key, "not used anymore.")
            del state_dict[key]

        # copy any newly defined parameters.
        for key in self.state_dict():
            if key not in state_dict:
                print("[INFO] adding", key)
                state_dict[key] = self.state_dict()[key]


# a dummy arch, we config the model.
@register_model_architecture("mmmodel", "mmarch")
def mmarch(args):
    pass
