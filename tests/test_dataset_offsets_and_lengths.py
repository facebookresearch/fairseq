import contextlib
import mmap
import os
import unittest
from io import BytesIO
from typing import Dict, Union
from unittest.mock import MagicMock, mock_open, patch

import numpy
import numpy as np
import soundfile

from fairseq.data import FileAudioDataset
from fairseq.data.audio.audio_utils import ParsedPath, parse_path
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform

SAMPLE_RATE = 16000

_soundfile_read = soundfile.read

TEST_SEQUENCE = np.linspace(-1, 1, 9999)


class TestParsePathOffsetsAndLengths(unittest.TestCase):
    def test_parse_path(self):
        test_cases = [
            (
                "wav file with no offset/length",
                "/path/to/myfile.wav",
                ParsedPath(
                    "/path/to/myfile.wav",
                    start=0,
                    frames=-1,
                    zip_offset=0,
                    zip_length=-1,
                ),
            ),
            (
                "wav file with offset & length",
                "/path/to/myfile.wav:123:456",
                ParsedPath(
                    "/path/to/myfile.wav",
                    start=123,
                    frames=456,
                    zip_offset=0,
                    zip_length=-1,
                ),
            ),
            (
                "zip file with no offset/length",
                "/path/to/myfile.zip",
                ParsedPath(
                    "/path/to/myfile.zip",
                    start=0,
                    frames=-1,
                    zip_offset=0,
                    zip_length=-1,
                ),
            ),
            (
                "zip file with offset & length",
                "/path/to/myfile.zip:123:456",
                ParsedPath(
                    "/path/to/myfile.zip",
                    start=0,
                    frames=-1,
                    zip_offset=123,
                    zip_length=456,
                ),
            ),
        ]

        for subtest, path, expected in test_cases:
            with self.subTest(subtest):
                with patch("os.path.isfile", return_value=True):
                    actual = parse_path(path)
                self.assertEqual(actual, expected)


class TestDatasetOffsetsAndLengths(unittest.TestCase):
    def test_file_audio_dataset_audio_with_offsets_etc(self):
        files = {
            "/path/to/myfile1.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile2.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile1.zip": encode_as_wav(TEST_SEQUENCE, pad_zip=123),
            "/path/to/myfile2.zip": encode_as_wav(TEST_SEQUENCE),
        }
        manifest_text = [
            "/dev/null",
            "/path/to/myfile1.wav:123:456\t456",
            "/path/to/myfile2.wav\t456",
            "/path/to/myfile1.zip:123:456\t456",
            "/path/to/myfile2.zip\t456",
        ]
        expected_data = [
            TEST_SEQUENCE[123 : 123 + 456],
            TEST_SEQUENCE,
            TEST_SEQUENCE[:206],
            TEST_SEQUENCE,
        ]

        files["/dev/null/train.tsv"] = os.linesep.join(manifest_text)

        with mock_files(files):
            dataset = FileAudioDataset("/dev/null/train.tsv", sample_rate=SAMPLE_RATE)
            assert len(expected_data) == len(dataset)
            for item, expected_sequence in zip(dataset, expected_data):
                filename = dataset.fnames[item["id"]]
                with self.subTest(filename):
                    numpy.testing.assert_almost_equal(
                        item["source"].numpy(), expected_sequence, decimal=4
                    )

    def test_file_audio_dataset_zip_with_offsets_etc(self):
        files = {
            "/path/to/myfile1.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile2.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile3.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile4.zip": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile5.zip": encode_as_wav(TEST_SEQUENCE, pad_zip=123),
        }
        manifest_text = [
            "/dev/null",
            "/path/to/myfile1.wav\t9999",
            "/path/to/myfile2.wav:123:456\t456",
            "/path/to/myfile3.wav:456:789\t789",
            "/path/to/myfile4.zip\t9999",
            "/path/to/myfile5.zip:123:456\t456",
        ]
        expected = [
            ("wav file, no offset, should return what we give it.", TEST_SEQUENCE),
            ("wav file with offset 123, length 456", TEST_SEQUENCE[123 : 123 + 456]),
            ("wav file with offset 456, length 789", TEST_SEQUENCE[456 : 456 + 789]),
            ("zip file with no offset/length", TEST_SEQUENCE),
            (
                "zip file provides *byte* offset & length. Will cut off wav files longer than that.",
                TEST_SEQUENCE[:206],
            ),
        ]

        files["/dev/null/train.tsv"] = os.linesep.join(manifest_text)

        with mock_files(files):
            dataset = FileAudioDataset("/dev/null/train.tsv", sample_rate=SAMPLE_RATE)
            self.assertEqual(len(dataset.fnames), len(expected))

            for item, (message, expected_sequence) in zip(dataset, expected):
                with self.subTest(message):
                    numpy.testing.assert_almost_equal(
                        item["source"].numpy().flatten(), expected_sequence, decimal=4
                    )

    def test_get_features_or_waveform(self):
        files = {
            "/path/to/myfile1.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile2.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile3.wav": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile4.zip": encode_as_wav(TEST_SEQUENCE),
            "/path/to/myfile5.zip": encode_as_wav(TEST_SEQUENCE, pad_zip=456),
        }
        test_cases = [
            (
                "wav file, no offset, should return what we give it.",
                "/path/to/myfile1.wav",
                TEST_SEQUENCE,
            ),
            (
                "wav file with offset 123, length 456",
                "/path/to/myfile2.wav:123:456",
                TEST_SEQUENCE[123 : 123 + 456],
            ),
            (
                "wav file with offset 456, length 789",
                "/path/to/myfile3.wav:456:789",
                TEST_SEQUENCE[456 : 456 + 789],
            ),
            (
                "zip file with no offset/length",
                "/path/to/myfile4.zip",
                TEST_SEQUENCE,
            ),
            (
                "zip file provides *byte* offset & length. Will cut off wav files longer than that.",
                "/path/to/myfile5.zip:456:123",
                TEST_SEQUENCE[:39],
            ),
        ]

        with mock_files(files):
            for message, filename_unparsed, expected_sequence in test_cases:
                with self.subTest(message):
                    features = get_features_or_waveform(
                        filename_unparsed, need_waveform=True
                    )
                    numpy.testing.assert_almost_equal(
                        expected_sequence, features.flatten(), decimal=4
                    )


def encode_as_wav(x, sample_rate=SAMPLE_RATE, pad_zip=0):
    bytes_io = BytesIO()
    soundfile.write(bytes_io, x, samplerate=sample_rate, format="wav")
    pad = bytes(ord("0") + (n % 10) for n in range(pad_zip))
    return pad + bytes_io.getvalue()


@contextlib.contextmanager
def mock_files(file_dict: Dict[str, Union[bytes, str]]):
    def builtins_open(file, mode="r", *args, **kwargs):
        data = file_dict[file]
        if "b" not in mode and isinstance(data, bytes):
            data = data.decode()
        mo = mock_open(read_data=data)
        mo.return_value.fileno.return_value = file
        return mo(file, mode, *args, **kwargs)

    def soundfile_read(file, *args, **kwargs):
        if isinstance(file, str) and file in file_dict:
            file = BytesIO(file_dict[file])
        return _soundfile_read(file, *args, **kwargs)

    def mmap_mmap(file, *args, **kwargs):
        return MagicMock(**{"__enter__.return_value": file_dict[file]})

    with patch(
        "soundfile.read", side_effect=soundfile_read
    ) as mock_soundfile_read, patch(
        "os.path.isfile", return_value=True
    ) as mock_isfile, patch(
        "builtins.open", side_effect=builtins_open
    ) as mock_builtins_open, patch(
        "mmap.mmap", side_effect=mmap_mmap
    ) as mock_mmap:

        assert open is mock_builtins_open
        assert os.path.isfile is mock_isfile
        assert soundfile.read is mock_soundfile_read
        assert mmap.mmap is mock_mmap
        yield


if __name__ == "__main__":
    unittest.main()
