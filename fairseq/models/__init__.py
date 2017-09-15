# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from . import fconv


__all__ = ['fconv']

arch_model_map = {}
for model in __all__:
    archs = locals()[model].get_archs()
    for arch in archs:
        assert arch not in arch_model_map, 'Duplicate model architecture detected: {}'.format(arch)
        arch_model_map[arch] = model
