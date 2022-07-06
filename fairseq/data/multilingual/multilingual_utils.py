# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Optional, Sequence

import torch

from fairseq.data import Dictionary

DATA_SOURCE_PREFIX_TAGS = {
    "mining": "<MINED_DATA>",
    "mmt_bt": "<MMT_BT_DATA>",
    "smt_bt": "<SMT_BT_DATA>",
}


class EncoderLangtok(Enum):
    """
    Prepend to the beginning of source sentence either the
    source or target language token. (src/tgt).
    """

    src = "src"
    tgt = "tgt"


class LangTokSpec(Enum):
    main = "main"
    mono_dae = "mono_dae"
    mono_lm = "mono_lm"
    mono_mixed_task = "mono_mixed_task"  # both of the above


class LangTokStyle(Enum):
    multilingual = "multilingual"
    mbart = "mbart"


@torch.jit.export
def get_lang_tok(
    lang: str, lang_tok_style: str, spec: str = LangTokSpec.main.value
) -> str:
    # TOKEN_STYLES can't be defined outside this fn since it needs to be
    # TorchScriptable.
    TOKEN_STYLES: Dict[str, str] = {
        LangTokStyle.mbart.value: "[{}]",
        LangTokStyle.multilingual.value: "__{}__",
    }

    if spec.endswith("dae"):
        lang = f"{lang}_dae"
    elif spec.endswith("lm"):
        lang = f"{lang}_lm"
    elif spec.endswith("mined"):
        lang = f"{lang}_mined"
    style = TOKEN_STYLES[lang_tok_style]
    return style.format(lang)


def augment_dictionary(
    dictionary: Dictionary,
    language_list: List[str],
    lang_tok_style: str,
    langtoks_specs: Sequence[str] = (LangTokSpec.main.value,),
    extra_data: Optional[Dict[str, str]] = None,
    add_data_source_prefix_tags: bool = False,
    add_ssl_task_tokens: bool = False,
    finetune_dict_specs: Optional[Dict[str, str]] = None,
) -> None:
    for spec in langtoks_specs:
        for language in language_list:
            dictionary.add_symbol(
                get_lang_tok(lang=language, lang_tok_style=lang_tok_style, spec=spec)
            )

    if (
        lang_tok_style == LangTokStyle.mbart.value
        or (
            extra_data is not None
            and (
                (LangTokSpec.mono_dae.value in extra_data)
                or (LangTokSpec.mono_mixed_task.value in extra_data)
            )
        )
        or (
            finetune_dict_specs is not None
            and LangTokSpec.mono_dae.value in finetune_dict_specs
        )
    ):
        dictionary.add_symbol("<mask>")
        if add_ssl_task_tokens:
            dictionary.add_symbol("__dae__")

    # Add special tokens.
    if add_data_source_prefix_tags:
        for name, tok in DATA_SOURCE_PREFIX_TAGS.items():
            dictionary.add_symbol(tok)

    if (
        extra_data is not None
        and (
            (LangTokSpec.mono_lm.value in extra_data)
            or (LangTokSpec.mono_mixed_task.value in extra_data)
        )
        or (
            finetune_dict_specs is not None
            and LangTokSpec.mono_lm.value in finetune_dict_specs
        )
    ):
        if add_ssl_task_tokens:
            dictionary.add_symbol("__lm__")
