# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

"""
A modified version of torch.cuda.nccl.all_reduce for launching kernels on each
GPU separately.
"""

import ctypes
import warnings

lib = None
_uid = None
_rank = None
_num_devices = None
_comm = None

__all__ = ['all_reduce', 'initialize', 'get_unique_id']


def _libnccl():
    global lib
    if lib is None:
        lib = ctypes.cdll.LoadLibrary(None)
        if hasattr(lib, 'ncclCommDestroy'):
            lib.ncclCommDestroy.restype = None
            lib.ncclGetErrorString.restype = ctypes.c_char_p
        else:
            lib = None
    return lib


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    if _libnccl() is None:
        warnings.warn('NCCL library not found. Check your LD_LIBRARY_PATH')
        return False

    return True


_communicators = {}

# ncclDataType_t
ncclChar = 0
ncclInt = 1
ncclHalf = 2
ncclFloat = 3
ncclDouble = 4
ncclInt64 = 5
ncclUint64 = 6

# ncclRedOp_t
SUM = 0
PROD = 1
MAX = 2
MIN = 3

nccl_types = {
    'torch.cuda.ByteTensor': ncclChar,
    'torch.cuda.CharTensor': ncclChar,
    'torch.cuda.IntTensor': ncclInt,
    'torch.cuda.HalfTensor': ncclHalf,
    'torch.cuda.FloatTensor': ncclFloat,
    'torch.cuda.DoubleTensor': ncclDouble,
    'torch.cuda.LongTensor': ncclInt64,
}


class NcclError(RuntimeError):
    def __init__(self, status):
        self.status = status
        msg = '{0} ({1})'.format(lib.ncclGetErrorString(status), status)
        super(NcclError, self).__init__(msg)


class NcclComm(ctypes.c_void_p):
    def __del__(self):
        lib.ncclCommDestroy(self)


class NcclUniqueId(ctypes.Structure):
    _fields_ = [
        ('internal', ctypes.c_uint8 * 128)
    ]


def check_error(status):
    if status != 0:
        raise NcclError(status)


_uids = []


def get_unique_id():
    if _libnccl() is None:
        raise RuntimeError('Unable to load NCCL library')

    uid = NcclUniqueId()
    check_error(lib.ncclGetUniqueId(ctypes.byref(uid)))
    _uids.append(uid)  # Don't allow UIDs to be collected
    return uid


def initialize(num_devices, uid, rank):
    global _num_devices, _uid, _rank

    if _libnccl() is None:
        raise RuntimeError('Unable to load NCCL library')

    _num_devices = num_devices
    if rank != 0:
        _uid = NcclUniqueId.from_buffer_copy(uid)
    else:
        _uid = uid
    _rank = rank


def communicator():
    global _comm
    if _uid is None:
        raise RuntimeError('NCCL not initialized')
    if _comm is None:
        comm = ctypes.c_void_p()
        check_error(lib.ncclCommInitRank(
            ctypes.byref(comm),
            ctypes.c_int(_num_devices),
            _uid,
            ctypes.c_int(_rank)))
        _comm = comm
    return _comm


def all_reduce(input, output=None, op=SUM, stream=None):
    comm = communicator()
    if output is None:
        output = input
    if stream is not None:
        stream = stream.cuda_stream
    data_type = nccl_types[input.type()]
    check_error(lib.ncclAllReduce(
        ctypes.c_void_p(input.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_size_t(input.numel()),
        data_type,
        op,
        comm,
        ctypes.c_void_p(stream)))
    return output
