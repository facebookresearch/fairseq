# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.data.encoders import register_bpe

SPACE = chr(32)
SPACE_ESCAPE = chr(9601)


@register_bpe('characters')
class Characters(object):
    def __init__(self, args):
        pass

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def encode(x: str) -> str:
        escaped = x.replace(SPACE, SPACE_ESCAPE)
        return SPACE.join(list(escaped))

    @staticmethod
    def decode(x: str) -> str:
        return x.replace(SPACE, '').replace(SPACE_ESCAPE, SPACE)
