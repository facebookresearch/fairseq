# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

try:
    import boto3
    from fairseq.fb_pathhandlers import S3PathHandler
except ImportError:
    boto3 = None
    S3PathHandler = None


# Hack to make the test cases ordered.
# https://stackoverflow.com/questions/4005695/changing-order-of-unit-tests-in-python
unittest.TestLoader.sortTestMethodsUsing = lambda _, x, y: y > x


@unittest.skipIf(not boto3, "Requires boto3 install")
class TestsS3(unittest.TestCase):
    s3_auth = False
    skip_s3_auth_required_tests_message = (
        "Provide an s3 project and bucket you are"
        + "authorised against, then set the s3_auth flag to True"
    )

    #############################################
    # Shared
    #############################################
    @classmethod
    def setUpClass(cls):
        # NOTE: user can change these locations.
        cls.s3_bucket = "fairusersglobal"
        cls.s3_rel_path = os.path.expandvars("users/$USER/private/home/$USER/.fairseq/test_s3_pathhandler")
        cls.s3_full_path = "s3://" + cls.s3_bucket + "/" + cls.s3_rel_path
        cls.s3_pathhandler = S3PathHandler()

    @classmethod
    def tearDownClass(cls, _s3_auth=s3_auth):
        if not _s3_auth:
            return

        # Recursive deletion is not implemented,
        # so let's delete each file and directory.

        # Delete all files
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir1", "f1_write_string"]))
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir1", "f2_write_bytes"]))
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir2", "f1_write_string_from_local"]))
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir2", "f2_write_bytes_from_local"]))
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir2", "f3_write_string_from_local"]))
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir2", "f4_write_bytes_from_local"]))

        # Delete all directories.
        cls.s3_pathhandler._rm("/".join([cls.s3_full_path, "dir3", "dir4/"]))
        for i in (1, 2, 3):
            cls.s3_pathhandler._rm("/".join([cls.s3_full_path, f"dir{i}/"]))

        assert cls.s3_pathhandler._ls(cls.s3_full_path) == []

    #############################################
    # Up here, test class attributes,
    # and helpers that don't require S3 access.
    #############################################

    def test_00_supported_prefixes(self):
        supported_prefixes = self.s3_pathhandler._get_supported_prefixes()
        self.assertEqual(supported_prefixes, ["s3://"])

    # # Require S3 Authentication ====>

    #############################################
    # Organization of s3 setup
    # dir1/
    #   f1 <- small (via open)
    #   f2 <- large checkpoint file (via open)
    # dir2/
    #   f3 <- small (via copy(), from dir1)
    #   f4 <- large checkpoint file (via copy_from_local)
    # dir3/
    #   dir4/
    #############################################

    #############################################
    # auth
    # Just check that client loads properly
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_01_add_client_to_handler(self):
        self.s3_pathhandler._get_client(
            "/".join([self.s3_full_path, "path", "file.txt"])
        )
        # self.assertTrue(isinstance(self.s3_pathhandler.client, botocore.client.S3)) # TODO

    # TODO: make sure that the error message displays properly if authentication is messed up.

    #############################################
    # mkdirs
    # Set up the dirs
    # (in BASE)
    #   +dir1
    #   +dir2
    #   +dir3
    #   +dir4
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_02_mkdirs_must_end_with_slash(self):
        with self.assertRaises(AssertionError):
            self.s3_pathhandler._mkdirs("/".join([self.s3_full_path, "fail"]))

    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_03_mkdirs(self):
        # dir{1,2,3} in BASE
        for i in (1, 2, 3):
            self.s3_pathhandler._mkdirs("/".join([self.s3_full_path, f"dir{i}/"]))

        # Make a nested directory in dir3
        self.s3_pathhandler._mkdirs("/".join([self.s3_full_path, "dir3/dir4/"]))

    #############################################
    # open (w/wb)
    #   +f1
    #   +f2
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_04_open_write_mode(self):
        with self.s3_pathhandler._open("/".join([self.s3_full_path, "dir1", "f1_write_string"]), 'w') as f:
            f.write("This is a test of writing a string.")

        with self.s3_pathhandler._open("/".join([self.s3_full_path, "dir1", "f2_write_bytes"]), 'wb') as f:
            f.write(b"This is a test of writing bytes.")

    #############################################
    # open (r/rb)
    #   read f1
    #   read f2
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_05_open_read_mode(self):
        with self.s3_pathhandler._open("/".join([self.s3_full_path, "dir1", "f1_write_string"]), 'r') as f:
            self.assertEqual(
                f.read(),
                "This is a test of writing a string."
            )

        with self.s3_pathhandler._open("/".join([self.s3_full_path, "dir1", "f2_write_bytes"]), 'rb') as f:
            self.assertEqual(
                f.read(),
                b"This is a test of writing bytes."
            )

    #############################################
    # isdir / isfile / exists
    #   test dir{1,2,3,4}
    #   test f{1,2}
    #   test nonexistants
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_06_exists(self):
        # Path does not exist (if file)
        self.assertFalse(self.s3_pathhandler._exists("/".join([self.s3_full_path, "dir1", "FAIL"])))
        # Path does not exist (if dir)
        self.assertFalse(self.s3_pathhandler._exists("/".join([self.s3_full_path, "FAIL/"])))
        # Path exists (is file)
        self.assertTrue(self.s3_pathhandler._exists("/".join([self.s3_full_path, "dir1", "f1_write_string"])))
        # Path exists (is dir)
        self.assertTrue(self.s3_pathhandler._exists("/".join([self.s3_full_path, "dir1/"])))

    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_07_isdir(self):
        # Path does not exist (if file)
        self.assertFalse(self.s3_pathhandler._isdir("/".join([self.s3_full_path, "dir1", "FAIL"])))
        # Path does not exist (if dir)
        self.assertFalse(self.s3_pathhandler._isdir("/".join([self.s3_full_path, "FAIL/"])))
        # Path exists (is file)
        self.assertFalse(self.s3_pathhandler._isdir("/".join([self.s3_full_path, "dir1", "f1_write_string"])))
        # Path exists (is dir)
        self.assertTrue(self.s3_pathhandler._isdir("/".join([self.s3_full_path, "dir1/"])))

    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_08_isfile(self):
        # Path does not exist (if file)
        self.assertFalse(self.s3_pathhandler._isfile("/".join([self.s3_full_path, "dir1", "FAIL"])))
        # Path does not exist (if dir)
        self.assertFalse(self.s3_pathhandler._isfile("/".join([self.s3_full_path, "FAIL/"])))
        # Path exists (is file)
        self.assertTrue(self.s3_pathhandler._isfile("/".join([self.s3_full_path, "dir1", "f1_write_string"])))
        # Path exists (is dir)
        self.assertFalse(self.s3_pathhandler._isfile("/".join([self.s3_full_path, "dir1/"])))

    #############################################
    # copy
    #   copy f1 -> f3
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_09_copy(self):
        self.assertTrue(
            self.s3_pathhandler._copy(
                "/".join([self.s3_full_path, "dir1", "f1_write_string"]),
                "/".join([self.s3_full_path, "dir2", "f3_write_string"])
            )
        )

    #############################################
    # ls
    #   ls dir{1,2,3,4}
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_10_ls(self):
        # Path does not exist (if file)
        self.assertEqual([], self.s3_pathhandler._ls("/".join([self.s3_full_path, "dir1", "FAIL"])))
        # Path does not exist (if dir)
        self.assertEqual([], self.s3_pathhandler._ls("/".join([self.s3_full_path, "FAIL/"])))
        # Path exists (is file)
        self.assertEqual(
            ["/".join([self.s3_rel_path, "dir1", "f1_write_string"])],
            self.s3_pathhandler._ls("/".join([self.s3_full_path, "dir1", "f1_write_string"]))
        )
        # Path exists (is dir)
        self.assertEqual(
            {
                "/".join([self.s3_rel_path, "dir1/"]),  # TODO: should the trailing slash be
                "/".join([self.s3_rel_path, "dir1", "f1_write_string"]),
                "/".join([self.s3_rel_path, "dir1", "f2_write_bytes"])
            },
            set(self.s3_pathhandler._ls("/".join([self.s3_full_path, "dir1/"])))
        )

    #############################################
    # rm
    #   rm f3
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_11_rm(self):
        path = "/".join([self.s3_full_path, "dir2", "f3_write_string"])
        self.assertTrue(self.s3_pathhandler._exists(path))
        self.assertTrue(self.s3_pathhandler._isfile(path))
        self.assertFalse(self.s3_pathhandler._isdir(path))

        self.s3_pathhandler._rm(path)

        self.assertFalse(self.s3_pathhandler._exists(path))
        self.assertFalse(self.s3_pathhandler._isfile(path))
        self.assertFalse(self.s3_pathhandler._isdir(path))

    #############################################
    # get_local_path
    #   Retrieve f{1,2}
    #   Check file contents.
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_12_get_local_path(self):
        s3_path_f1 = "/".join([self.s3_full_path, "dir1", "f1_write_string"])
        s3_path_f2 = "/".join([self.s3_full_path, "dir1", "f2_write_bytes"])

        local_path_f1 = self.s3_pathhandler._get_local_path(s3_path_f1)
        local_path_f2 = self.s3_pathhandler._get_local_path(s3_path_f2)

        with open(local_path_f1, 'r') as f:
            self.assertEqual(
                f.read(),
                "This is a test of writing a string."
            )
        with open(local_path_f2, 'rb') as f:
            self.assertEqual(
                f.read(),
                b"This is a test of writing bytes."
            )

    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_13_get_local_path_idempotent(self):
        """
        Call _get_local_path multiple times.
        Check that we keep returning the same cached copy instead of redownloading.
        """
        s3_path_f1 = "/".join([self.s3_full_path, "dir1", "f1_write_string"])

        REPEATS = 3
        local_paths = [self.s3_pathhandler._get_local_path(s3_path_f1) for _ in range(REPEATS)]
        for local_path in local_paths[1:]:
            self.assertEqual(local_path, local_paths[0])

        with open(local_paths[0], 'r') as f:
            self.assertEqual(
                f.read(),
                "This is a test of writing a string."
            )

    # TODO: make sure it fails if asked for a directory
    # TODO: make sure that the returned path is appropriately placed.

    ##############################################
    # copy_from_local
    #   Upload local copies of f1, f2 -> f3, f4.
    # Check contents via open(), and via another get_local_path
    ##############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_14_copy_from_local(self):
        s3_src_path_f1 = "/".join([self.s3_full_path, "dir1", "f1_write_string"])
        s3_src_path_f2 = "/".join([self.s3_full_path, "dir1", "f2_write_bytes"])

        local_path_f1 = self.s3_pathhandler._get_local_path(s3_src_path_f1)
        local_path_f2 = self.s3_pathhandler._get_local_path(s3_src_path_f2)

        s3_dst_path_f1 = "/".join([self.s3_full_path, "dir2", "f1_write_string_from_local"])
        s3_dst_path_f2 = "/".join([self.s3_full_path, "dir2", "f2_write_bytes_from_local"])

        self.assertTrue(self.s3_pathhandler._copy_from_local(local_path_f1, s3_dst_path_f1))
        self.assertTrue(self.s3_pathhandler._copy_from_local(local_path_f2, s3_dst_path_f2))

    #############################################
    # symlink
    #   should fail
    #############################################
    @unittest.skipIf(not s3_auth, skip_s3_auth_required_tests_message)
    def test_15_symlink(self):
        s3_src_path_f1 = "/".join([self.s3_full_path, "dir1", "f1_write_string"])
        s3_dst_path_f1 = "/".join([self.s3_full_path, "dir2", "f1_write_string_symlink"])
        with self.assertRaises(NotImplementedError):
            self.s3_pathhandler._symlink(s3_src_path_f1, s3_dst_path_f1)
