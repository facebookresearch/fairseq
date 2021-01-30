# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from fairseq.data.encoders import register_bpe
from fairseq.data.encoders.byte_utils import (
    SPACE,
    SPACE_ESCAPE,
    byte_encode,
    smart_byte_decode,
)


@register_bpe("bytes")
class Bytes(object):
    def __init__(self, *unused):
        pass

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def encode(x: str) -> str:
        encoded = byte_encode(x)
        escaped = encoded.replace(SPACE, SPACE_ESCAPE)
        return SPACE.join(list(escaped))

    @staticmethod
    def decode(x: str) -> str:
        unescaped = x.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)
        return smart_byte_decode(unescaped)
