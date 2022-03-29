# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .task import *
from .vlmtask import *
from .retritask import *

try:
    from .fairseqmmtask import *
except ImportError:
    pass

try:
    from .milncetask import *
except ImportError:
    pass

try:
    from .expretritask import *
except ImportError:
    pass
