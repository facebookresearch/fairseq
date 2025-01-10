#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""
user_input_name = str(input())
print("Hey, " + user_input_name + ". Thank you for choosing us")

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    cli_main()
 else raise Excpetion('Function/program failed, if youu would like to continue, rerun the program')
