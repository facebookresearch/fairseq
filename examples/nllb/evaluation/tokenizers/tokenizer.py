# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Dependencies:
    QCRI for tokenizing Arabic in IWSLT
'''
QCRI = "../qcri_arabic_normalizer3.0/"
QCRI_SCRIPT = "{QCRI}/qcri_normalizer_mada3.2_aramorph1.2.1.sh"
KYTEA = "../kytea-0.4.7/src/bin/kytea"
KYTEA_MODEL = "../kytea-0.4.7/data/model.bin"
KMSEG = "../kmseg.py"
MYSEG = "../myseg.py"
NORMALIZE_ROM = "../romanian/normalize_rom.py"
REMOVE_DIACRITICS = "../romanian/remove_diacritics.py"
IndicNLP_PREPROCESS = "../indicTrans/scripts/preprocess_translate.py"


def tokenize_13a(args, src, tgt, ref, hyp):
    sacrebleu_cmd = f"-tok {'zh' if tgt == 'zho_Hans' else '13a'}"
    return ref, hyp, "", sacrebleu_cmd


def tokenize_indic_nlp(args, src, tgt, ref, hyp):
    def convert_lang(lang):
        if lang == 'hin_Deva':
            return 'hi'
        elif lang == "mar_Deva":
            return 'mr'
        elif lang == "mal_Mlym":
            return 'ml'
        elif lang == 'tel_Telu':
            return 'te'
        elif lang == 'tam_Taml':
            return 'ta'
        elif lang == 'ben_Beng':
            return 'bn'
        elif lang == 'ory_Orya':
            return 'or'
        elif lang == 'kan_Knda':
            return 'kn'
        elif lang == 'guj_Gujr':
            return 'gu'
        elif lang == "asm_Beng":
            return 'as'
        elif lang == "pan_Guru":
            return 'pa'
        else:
            raise ValueError(f'Unknown language {lang}')

    script = IndicNLP_PREPROCESS
    # tokenize the hypotheses and the references
    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"python {script} {ref} {tok_ref} {convert_lang(tgt)}",
        f"python {script} {hyp} {tok_hyp} {convert_lang(tgt)}"
    ])

    sacrebleu_cmd = "-tok none"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd


def tokenize_myseg(args, src, tgt, ref, hyp):
    script = MYSEG
    # tokenize the hypotheses and the references
    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"python2 {script} < {ref}  > {tok_ref}",
        f"python2 {script} < {hyp}  > {tok_hyp}",
    ])
    sacrebleu_cmd = "-tok none"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd

def tokenize_kmseg(args, src, tgt, ref, hyp):
    script = KMSEG
    # tokenize the hypotheses and the references
    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"python2 {script} < {ref}  > {tok_ref}",
        f"python2 {script} < {hyp}  > {tok_hyp}",
    ])
    sacrebleu_cmd = "-tok none"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd


def tokenize_romanian_sennrich(args, src, tgt, ref, hyp):
    
    # tokenize the hypotheses and the references
    norm_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.norm"
    norm_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.norm"

    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"python {NORMALIZE_ROM} < {ref}  > {norm_ref}",
        f"python {NORMALIZE_ROM} < {hyp}  > {norm_hyp}",
        f"python {REMOVE_DIACRITICS} < {norm_ref}  > {tok_ref}",
        f"python {REMOVE_DIACRITICS} < {norm_hyp}  > {tok_hyp}",
    ])
    sacrebleu_cmd = "-tok 13a"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd


def tokenize_qcri(args, src, tgt, ref, hyp):
    script = QCRI_SCRIPT
    # tokenize the hypotheses and the references
    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"cd {QCRI}",
        f"{script} {ref}  {tok_ref}",
        f"{script} {hyp}  {tok_hyp}",
    ])
    sacrebleu_cmd = "-tok none"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd


def tokenize_mecab(args, src, tgt, ref, hyp):
    sacrebleu_cmd = "-tok ko-mecab"
    return ref, hyp, "", sacrebleu_cmd


def tokenize_kytea(args, src, tgt, ref, hyp):
    # tokenize the hypotheses and the references
    tok_ref = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.ref.tok"
    tok_hyp = f"{args.output_dir}/{args.corpus}-{src}-{tgt}.hyp.tok"

    prep_cmd = '\n'.join([
        f"{KYTEA} -model {KYTEA_MODEL} < {ref} > {tok_ref}",
        f"{KYTEA} -model {KYTEA_MODEL} < {hyp} > {tok_hyp}",
    ])
    sacrebleu_cmd = "-tok none"
    return tok_ref, tok_hyp, prep_cmd, sacrebleu_cmd


def get_special_tokenizer(corpus, tgt):
    if tgt == 'eng_Latn':
        return tokenize_13a
    if corpus == 'flores101_indic':
        return tokenize_indic_nlp
    elif corpus == 'iwslt':
        if tgt == 'arb_Arab':
            return tokenize_qcri
        elif tgt == 'kor_Hang':
            return tokenize_mecab
        elif tgt == 'jpn_Jpan':
            return tokenize_kytea
    elif corpus == 'wat':
        if tgt == 'mya_Mymr':
            return tokenize_myseg
        elif tgt == 'khm_Khmr':
            return tokenize_kmseg
        elif tgt == 'hin_Deva':
            return tokenize_indic_nlp
    elif corpus == 'wmt':
        if tgt == 'ron_Latn':
            return tokenize_romanian_sennrich
    return tokenize_13a


def tokenize(args, src, tgt, ref, hyp):
    special_tokenizer = get_special_tokenizer(args.corpus, tgt)
    return special_tokenizer(args, src, tgt, ref, hyp)
