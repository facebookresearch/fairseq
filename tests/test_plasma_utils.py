import contextlib
import unittest
import tempfile
from io import StringIO

import numpy as np

from tests.test_binaries import train_language_model
from tests.utils import create_dummy_data, preprocess_lm_data

try:
    from pyarrow import plasma
    from fairseq.data.plasma_utils import PlasmaView, PlasmaStore

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

dummy_path = 'dummy'


@unittest.skipUnless(PYARROW_AVAILABLE, "")
class TestPlasmaView(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_file = tempfile.NamedTemporaryFile()  # noqa: P201
        self.path = self.tmp_file.name
        self.server = PlasmaStore.start(path=self.path)
        self.client = plasma.connect(self.path, num_retries=10)

    def tearDown(self) -> None:
        self.client.disconnect()
        self.tmp_file.close()
        self.server.kill()

    def test_two_servers_do_not_share_object_id_space(self):
        data_server_1 = np.array([0, 1])
        data_server_2 = np.array([2, 3])
        server_2_path = self.path
        with tempfile.NamedTemporaryFile() as server_1_path:
            server = PlasmaStore.start(path=server_1_path.name, nbytes=10000)
            arr1 = PlasmaView(
                data_server_1, dummy_path, 1, plasma_path=server_1_path.name
            )
            assert len(arr1.client.list()) == 1
            assert (arr1.array == data_server_1).all()
            arr2 = PlasmaView(data_server_2, dummy_path, 1, plasma_path=server_2_path)
            assert (arr2.array == data_server_2).all()
            assert (arr1.array == data_server_1).all()
            server.kill()

    def test_hash_collision(self):
        data_server_1 = np.array([0, 1])
        data_server_2 = np.array([2, 3])
        arr1 = PlasmaView(data_server_1, dummy_path, 1, plasma_path=self.path)
        assert len(arr1.client.list()) == 1
        arr2 = PlasmaView(data_server_2, dummy_path, 1, plasma_path=self.path)
        assert len(arr1.client.list()) == 1
        assert len(arr2.client.list()) == 1
        assert (arr2.array == data_server_1).all()
        # New hash key based on tuples
        arr3 = PlasmaView(
            data_server_2, dummy_path, (1, 12312312312, None), plasma_path=self.path
        )
        assert (
            len(arr2.client.list()) == 2
        ), "No new object was created by using a novel hash key"
        assert (
            arr3.object_id in arr2.client.list()
        ), "No new object was created by using a novel hash key"
        assert (
            arr3.object_id in arr3.client.list()
        ), "No new object was created by using a novel hash key"
        del arr3, arr2, arr1

    @staticmethod
    def _assert_view_equal(pv1, pv2):
        np.testing.assert_array_equal(pv1.array, pv2.array)

    def test_putting_same_array_twice(self):
        data = np.array([4, 4, 4])
        arr1 = PlasmaView(data, dummy_path, 1, plasma_path=self.path)
        assert len(self.client.list()) == 1
        arr1b = PlasmaView(
            data, dummy_path, 1, plasma_path=self.path
        )  # should not change contents of store
        arr1c = PlasmaView(
            None, dummy_path, 1, plasma_path=self.path
        )  # should not change contents of store

        assert len(self.client.list()) == 1
        self._assert_view_equal(arr1, arr1b)
        self._assert_view_equal(arr1, arr1c)
        PlasmaView(
            data, dummy_path, 2, plasma_path=self.path
        )  # new object id, adds new entry
        assert len(self.client.list()) == 2

        new_client = plasma.connect(self.path)
        assert len(new_client.list()) == 2  # new client can access same objects
        assert isinstance(arr1.object_id, plasma.ObjectID)
        del arr1b
        del arr1c

    def test_plasma_store_full_raises(self):
        with tempfile.NamedTemporaryFile() as new_path:
            server = PlasmaStore.start(path=new_path.name, nbytes=10000)
            with self.assertRaises(plasma.PlasmaStoreFull):
                # 2000 floats is more than 2000 bytes
                PlasmaView(
                    np.random.rand(10000, 1), dummy_path, 1, plasma_path=new_path.name
                )
            server.kill()

    def test_object_id_overflow(self):
        PlasmaView.get_object_id("", 2 ** 21)

    def test_training_lm_plasma(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer_lm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_language_model(
                    data_dir,
                    "transformer_lm",
                    ["--use-plasma-view", "--plasma-path", self.path],
                    run_validation=True,
                )
