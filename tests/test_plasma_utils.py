import unittest
import tempfile
import numpy as np


try:
    from pyarrow import plasma
    from fairseq.data import TokenBlockDataset
    from fairseq.data.plasma_utils import PlasmaView, start_plasma_store

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


@unittest.skipUnless(PYARROW_AVAILABLE, "")
class TestPlasmaView(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_file = tempfile.NamedTemporaryFile()
        self.path = self.tmp_file.name
        self.server = start_plasma_store(path=self.tmp_file.name)
        self.client = plasma.connect(self.path, num_retries=10)

    def tearDown(self) -> None:
        self.tmp_file.close()
        self.server.kill()

    def test_putting_twice(self):
        data = np.array([4, 4, 4,])
        arr1 = PlasmaView(data, 1, path=self.path)
        assert len(self.client.list()) == 1
        arr1b = PlasmaView(
            data, 1, path=self.path
        )  # should not change contents of store
        assert len(self.client.list()) == 1
        assert (arr1.array == arr1b.array).all()

        PlasmaView(data, 2, path=self.path)  # new object id, adds new entry
        assert len(self.client.list()) == 2

        new_client = plasma.connect(self.path)
        assert len(new_client.list()) == 2  # new client can access same objects
        assert isinstance(arr1.object_id, plasma.ObjectID)

    def test_underallocation_raises(self):
        new_path = tempfile.NamedTemporaryFile()
        server = start_plasma_store(path=new_path.name, nbytes=10000)
        with self.assertRaises(plasma.PlasmaStoreFull):
            # 2000 floats is more than 2000 bytes
            PlasmaView(np.random.rand(10000, 1), 1, path=new_path.name)
        server.kill()

    def test_object_id_overflow(self):
        PlasmaView.get_object_id(2 ** 21)

    def test_object_id_arr_handles_billion_tokens(self):
        arr = np.arange(1000000000).reshape(4,-1)
        import time
        t0 = time.time()
        a2 = PlasmaView.get_object_id_arr_unused(arr)
        tbig = time.time() - t0
        t1 = time.time()
        a1 = PlasmaView.get_object_id_arr_unused(arr[:100,0])
        tsmall = time.time() - t1
        assert tbig / tsmall < 2.


    def test_object_id_same_data_diff_shape(self):
        a1 = np.arange(12)
        h1 = PlasmaView.get_object_id_arr_unused(a1)
        a2 = a1.reshape(3, 2, 2)
        h2 = PlasmaView.get_object_id_arr_unused(a2)
        assert h1 != h2
