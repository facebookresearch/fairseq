# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import logging


try:
    import pyarrow.plasma as plasma

    PYARROW_AVAILABLE = True
except ImportError:
    plasma = None
    PYARROW_AVAILABLE = False

import tempfile
import hashlib
from filelock import FileLock

NO_LOCK = os.getenv("NO_LOCK", False)
#logging.getLogger("filelock").setLevel(logging.ERROR)


class PlasmaArray:
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    """

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.disable = array.nbytes < 134217728  # disable for arrays <128MB
        self.object_id = None
        self.path = None

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None
        self._plasma = None

    @property
    def plasma(self):
        if self._plasma is None and not self.disable:
            self._plasma = plasma
        return self._plasma

    def start_server(self):
        if self.plasma is None or self._server is not None:
            return
        assert self.object_id is None
        assert self.path is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen(
            ["plasma_store", "-m", str(int(1.05 * self.array.nbytes)), "-s", self.path,]
        )

    def client(self):
        if self._client is None:
            self._client = self.plasma.connect(self.path, num_retries=200)
        return self._client

    def __getstate__(self):
        if self.plasma is None:
            return self.__dict__
        if self.object_id is None:
            self.start_server()
            self.object_id = self.client.put(self.array)
        state = self.__dict__.copy()
        del state["array"]
        state["_client"] = None
        state["_server"] = None
        state["_server_tmp"] = None
        state["_plasma"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.plasma is None:
            return
        self.array = self.client.get(self.object_id)

    def __del__(self):
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None
            # self._client.disconnect()


DEFAULT_PLASMA_PATH = "/tmp/plasma"

def plasma_store_contains(split_path, object_id, timeout_ms=30000) -> bool:
    id = PlasmaView.get_object_id(split_path, object_id)
    client = plasma.connect(DEFAULT_PLASMA_PATH, num_retries=200)
    ret = client.get(id, timeout_ms=timeout_ms)
    client.disconnect()
    return not isinstance(ret, plasma.ObjectNotAvailable)


class PlasmaView:
    def __init__(
        self, array, split_path: str, object_num: int, path=DEFAULT_PLASMA_PATH
    ):
        assert PYARROW_AVAILABLE
        assert split_path is not None

        self.path = path
        self.split_path = split_path
        self.object_id = self.get_object_id(self.split_path, object_num)
        self._client = None  # Initialize lazily for pickle, (TODO(SS): needed?)
        self._n = None
        self.use_lock = not NO_LOCK
        if array is not None:
            try:
                self.client.put(array, object_id=self.object_id)
                self.log("PUT")
            except plasma.PlasmaObjectExists:
                self.log("PlasmaObjectExists")

    @property
    def client(self):
        if self._client is None:
            self._client = plasma.connect(self.path, num_retries=200)
        return self._client
        # return self._client

    @property
    def array(self):
        """Fetch a read only view of a np array, stored in plasma."""
        self.log("GET")
        if self.use_lock:
            with FileLock("/tmp/plasma_read_lock"):
                ret = self.client.get(self.object_id)

        else:
            ret = self.client.get(self.object_id)
        self.log("GOT")
        return ret

    def log(self, msg: str) -> None:
        # print(f"pid: {os.getpid()}, id: {self.object_id}, lock: {self.use_lock}: {preamble}")
        pass

    @staticmethod
    def int_to_bytes(x: int) -> bytes:
        return x.to_bytes(
            (x.bit_length() + 7) // 8, "big"
        )  # https://tinyurl.com/56j5964v

    @staticmethod
    def get_object_id(split_path: str, object_num: int) -> plasma.ObjectID:
        hash = hashlib.blake2b(bytes(split_path, "utf-8"), digest_size=20)
        hash.update(object_num.to_bytes(4, byteorder="big"))
        return plasma.ObjectID(hash.digest())

    @staticmethod
    def get_object_id_arr_unused(arr) -> plasma.ObjectID:
        """Just hash the shape"""
        # TODO(SS): delete if useless
        hash = hashlib.blake2b(b"0", digest_size=20)
        for dim in arr.shape:
            hash.update(dim.to_bytes(4, byteorder="big"))
        return plasma.ObjectID(hash.digest())

    def __getstate__(self):
        """Called on pickle save, I believe"""
        self.client.disconnect()
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        """Called on pickle load, I believe"""
        self.__dict__.update(state)
        # self.client = plasma.connect(self.path, num_retries=200)

    def __del__(self):
        self._client = None

    def __len__(self):
        """Save reads by caching len"""
        if self._n is None:
            self._n = len(self.array)
        return self._n


ONE_TB = 1024 ** 4
GB200 = (1024 ** 3) * 100


def start_plasma_store(
    path=DEFAULT_PLASMA_PATH, nbytes: int = GB200
) -> subprocess.Popen:
    if not PYARROW_AVAILABLE:
        raise ImportError("please run pip install pyarrow to use --use_plasma_view")
    # best practice is to allocate more space than we need. The limitation seems to be the size of /dev/shm
    _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])
    plasma.connect(path, num_retries=200)  # If we can't connect we fail immediately
    return _server
