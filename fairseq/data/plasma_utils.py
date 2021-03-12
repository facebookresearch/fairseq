# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import subprocess
import json
import tempfile
import hashlib
from typing import Hashable

try:
    import pyarrow.plasma as plasma

    PYARROW_AVAILABLE = True
except ImportError:
    plasma = None
    PYARROW_AVAILABLE = False


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
            ["plasma_store", "-m", str(int(1.05 * self.array.nbytes)), "-s", self.path]
        )

    @property
    def client(self):
        if self._client is None:
            assert self.path is not None
            self._client = self.plasma.connect(self.path, num_retries=200)
        return self._client

    def __getstate__(self):
        """Called on pickle load"""
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
        """Called on pickle save"""
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


DEFAULT_PLASMA_PATH = "/tmp/plasma"


class PlasmaView:
    """Interface to write and read from shared memory. Whereas PlasmaArray writes to plasma on serialization,
    PlasmaView writes to shared memory on instantiation."""

    def __init__(self, array, split_path: str, hash_data: Hashable, plasma_path=None):
        """
        Args:
            array: numpy array to store. This can be read with ``PlasmaView().array``
            split_path: the path whence the data was read, used for hashing
            hash_data: other metadata about the array that can be used to create a unique key.
                as of writing, the 3 callers in ``TokenBlockDataset`` use::

                    hash_data = ((block_size, document_sep_len, str(break_mode), len(dataset)), 0|1|2)


        """
        assert PYARROW_AVAILABLE
        assert split_path is not None
        if plasma_path is None:
            plasma_path = DEFAULT_PLASMA_PATH

        self.path = plasma_path
        self.split_path = split_path
        self._client = None  # Initialize lazily for pickle. plasma clients should not be deep copied or serialized.
        self._n = None

        self.object_id = self.get_object_id(self.split_path, hash_data)
        try:
            self.client.put(array, object_id=self.object_id)
        except plasma.PlasmaObjectExists:
            pass

    @property
    def client(self):
        if self._client is None:
            self._client = plasma.connect(self.path, num_retries=200)
        return self._client

    @property
    def array(self):
        """Fetch a read only view of an np.array, stored in plasma."""
        ret = self.client.get(self.object_id)
        return ret

    @staticmethod
    def get_object_id(split_path: str, hash_data: Hashable):
        """Returns plasma.ObjectID from hashing split_path and object_num."""
        hash = hashlib.blake2b(bytes(split_path, "utf-8"), digest_size=20)
        harg = json.dumps(hash_data).encode("utf-8")
        hash.update(harg)
        return plasma.ObjectID(hash.digest())

    def __getstate__(self):
        """Called on pickle save"""
        self.disconnect()
        state = self.__dict__.copy()
        assert state["_client"] is None
        assert "object_id" in state
        return state

    def __setstate__(self, state):
        """Called on pickle load"""
        self.__dict__.update(state)

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self._client is not None:
            self._client.disconnect()
            self._client = None

    def __len__(self):
        """Save reads by caching len"""
        if self._n is None:
            self._n = len(self.array)
        return self._n


GB100 = (1024 ** 3) * 100


class PlasmaStore:
    def __init__(self, path=DEFAULT_PLASMA_PATH, nbytes: int = GB100):

        self.server = self.start(path, nbytes)

    def __del__(self):
        self.server.kill()

    @staticmethod
    def start(path=DEFAULT_PLASMA_PATH, nbytes: int = GB100) -> subprocess.Popen:
        if not PYARROW_AVAILABLE:
            raise ImportError("please run pip install pyarrow to use --use_plasma_view")
        # best practice is to allocate more space than we need. The limitation seems to be the size of /dev/shm
        _server = subprocess.Popen(["plasma_store", "-m", str(nbytes), "-s", path])
        plasma.connect(path, num_retries=200)  # If we can't connect we fail immediately
        return _server
