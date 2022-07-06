# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Use with get_train_log_metrics.py to extract per-language best steps. Example:
> python examples/nllb/modeling/scripts/get_train_log_metrics.py --filepath $FILEPATH \
    --pattern valid_main --metric ppl --print-steps --src --tgt |
    python examples/nllb/modeling/scripts/best_steps_lang_pair.py
[
    (15000, 'valid_main:fon-eng_ppl'), (20000, 'valid_main:eng-wol_ppl'),
    (20000, 'valid_main:kon-eng_ppl'), (25000, 'valid_main:eng-fuv_ppl'),
    (25000, ...
"""

import sys
from ast import literal_eval

if __name__ == "__main__":
    d = {}
    for line in sys.stdin:
        key, val = line.strip().split("\t")
        val2 = [(float(v), int(k)) for k, v in literal_eval(val).items()]
        d[key] = min(val2, key=lambda x: (x[0], -x[1]))

    print(sorted([(v[1], k) for k, v in d.items()]))
