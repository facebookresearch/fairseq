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
from ctypes.util import find_library

lib = None
nccl_2_0 = None
_uid = None
_rank = None
_num_devices = None
_comm = None

__all__ = ['all_reduce', 'initialize', 'get_unique_id']

# ncclDataType_t
nccl_types = {
    'torch.cuda.ByteTensor': 0,
    'torch.cuda.CharTensor': 0,
    'torch.cuda.IntTensor': 1,
    'torch.cuda.HalfTensor': 2,
    'torch.cuda.FloatTensor': 3,
    'torch.cuda.DoubleTensor': 4,
    'torch.cuda.LongTensor': 5,
}
nccl_types_2_0 = {
    'torch.cuda.ByteTensor': 0,
    'torch.cuda.CharTensor': 0,
    'torch.cuda.IntTensor': 2,
    'torch.cuda.HalfTensor': 6,
    'torch.cuda.FloatTensor': 7,
    'torch.cuda.DoubleTensor': 8,
    'torch.cuda.LongTensor': 4,
}

# ncclRedOp_t
SUM = 0
PROD = 1
MAX = 2
MIN = 3

status_codes_2_0 = {
    0: "Success",
    1: "Unhandled Cuda Error",
    2: "System Error",
    3: "Internal Error",
    4: "Invalid Argument Error",
    5: "Invalid Usage Error",
}

status_codes = {
    0: "Success",
    1: "Unhandled Cuda Error",
    2: "System Error",
    3: "Internal Error",
    4: "Invalid Device Pointer",
    5: "Invalid Rank",
    6: "Unsupported Device Count",
    7: "Device Not Found",
    8: "Invalid Device Index",
    9: "Lib Wrapper Not Set",
    10: "Cuda Malloc Failed",
    11: "Rank Mismatch",
    12: "Invalid Argument",
    13: "Invalid Type",
    14: "Invalid Operation",
}


def _libnccl():
    global nccl_2_0
    global lib
    global status_codes
    global nccl_types
    if lib is None:
        lib = ctypes.pydll.LoadLibrary(find_library('nccl'))
        if hasattr(lib, 'ncclCommDestroy'):
            lib.ncclCommDestroy.restype = None
        else:
            lib = None
        if hasattr(lib, 'ncclGroupStart'):
            nccl_2_0 = True
            status_codes = status_codes_2_0
            nccl_types = nccl_types_2_0
    return lib


class NcclError(RuntimeError):

    def __init__(self, status):
        self.status = status
        msg = '{0} ({1})'.format(status_codes.get(status), status)
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
    if _libnccl() is None:
        raise RuntimeError('Unable to load NCCL library')
    if _uid is None:
        raise RuntimeError('NCCL not initialized')
    if _comm is None:
        comm = NcclComm()
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
