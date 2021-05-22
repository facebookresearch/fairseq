# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

src_ckpt = "/checkpoint/wnhsu/w2v/archived/hubert_base_ls960_it2.pt"
ref_ckpt = "/checkpoint/wnhsu/w2v/hubert_icassp_oss_v3/iter2_km100-400k-grp-L6/oss.km500_p0_1_s334.pmw1_0.puw0_0.grpnorm.ml10.mp0_8.untie.mxsz250000.ufreq1.maxtok1400000.MU100k.s1337.ngpu32/checkpoint_last.pt"
new_ckpt = "/checkpoint/wnhsu/w2v/archived/hubert_base_ls960_it2_updated.pt"


def update_state(state):
    state["model"]["label_embs_concat"] = state["model"].pop("label_embs")
    state["args"].task = "hubert_pretraining"
    state["args"].labels = f"['{state['args'].labels}']"
    return state


src_state = torch.load(src_ckpt)
src_state = update_state(src_state)
torch.save(src_state, new_ckpt)
