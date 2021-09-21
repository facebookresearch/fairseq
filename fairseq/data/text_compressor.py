# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class TextCompressionLevel(Enum):
    none = 0
    low = 1
    high = 2


class TextCompressor(object):
    def __init__(
            self, level: TextCompressionLevel,
            max_input_byte_length: int = 2 ** 16
    ):
        self.level = level
        self.max_input_length = max_input_byte_length

    def compress(self, text: str) -> bytes:
        if self.level == TextCompressionLevel.low:
            import zlib
            # zlib: built-in, fast
            return zlib.compress(text.encode(), level=0)
        elif self.level == TextCompressionLevel.high:
            try:
                import unishox2
                # unishox2: optimized for short text but slower
            except ImportError:
                raise ImportError(
                    "Please install unishox2 for the text compression feature: "
                    "pip install unishox2-py3"
                )
            assert len(text.encode()) <= self.max_input_length
            return unishox2.compress(text)[0]
        else:
            return text.encode()

    def decompress(self, compressed: bytes) -> str:
        if self.level == TextCompressionLevel.low:
            import zlib
            return zlib.decompress(compressed).decode()
        elif self.level == TextCompressionLevel.high:
            try:
                import unishox2
            except ImportError:
                raise ImportError(
                    "Please install unishox2 for the text compression feature: "
                    "pip install unishox2-py3"
                )
            return unishox2.decompress(compressed, self.max_input_length)
        else:
            return compressed.decode()
