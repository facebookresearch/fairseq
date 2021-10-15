# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp


def _safe_readline(fd) -> str:
    pos = fd.tell()
    while True:
        try:
            return fd.readline()
        except UnicodeDecodeError:
            pos -= 1
            fd.seek(pos)  # search where this character begins


def find_offsets(filename: str, num_chunks: int) -> tp.List[int]:
    """
    given a file and a number of chuncks, find the offsets in the file
    to be able to chunk around full lines.
    """
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            _safe_readline(f)
            offsets[i] = f.tell()
        offsets[-1] = size
        return offsets


class ChunkLineIterator:
    """
    Iterator to properly iterate over lines of a file chunck.
    """

    def __init__(self, fd, start_offset: int, end_offset: int):
        self._fd = fd
        self._start_offset = start_offset
        self._end_offset = end_offset

    def __iter__(self) -> tp.Iterable[str]:
        self._fd.seek(self._start_offset)
        # next(f) breaks f.tell(), hence readline() must be used
        line = _safe_readline(self._fd)
        while line:
            pos = self._fd.tell()
            # f.tell() does not always give the byte position in the file
            # sometimes it skips to a very large number
            # it is unlikely that through a normal read we go from
            # end bytes to end + 2**32 bytes (4 GB) and this makes it unlikely
            # that the procedure breaks by the undeterministic behavior of
            # f.tell()
            if (
                self._end_offset > 0
                and pos > self._end_offset
                and pos < self._end_offset + 2 ** 32
            ):
                break
            yield line
            line = self._fd.readline()


class Chunker:
    """
    contextmanager to read a chunck of a file line by line.
    """

    def __init__(self, path: str, start_offset: int, end_offset: int):
        self.path = path
        self.start_offset = start_offset
        self.end_offset = end_offset

    def __enter__(self) -> ChunkLineIterator:
        self.fd = open(self.path, "r", encoding="utf-8")
        return ChunkLineIterator(self.fd, self.start_offset, self.end_offset)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.fd.close()
