#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
import logging
from omegaconf import MISSING, OmegaConf
import os
import os.path as osp
from pathlib import Path
import subprocess
from typing import Optional

from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import FairseqDataclass

script_dir = Path(__file__).resolve().parent
config_path = script_dir / "config"


logger = logging.getLogger(__name__)


@dataclass
class KaldiInitializerConfig(FairseqDataclass):
    data_dir: str = MISSING
    fst_dir: Optional[str] = None
    in_labels: str = MISSING
    out_labels: Optional[str] = None
    wav2letter_lexicon: Optional[str] = None
    lm_arpa: str = MISSING
    kaldi_root: str = MISSING
    blank_symbol: str = "<s>"
    silence_symbol: Optional[str] = None


def create_units(fst_dir: Path, in_labels: str, vocab: Dictionary) -> Path:
    in_units_file = fst_dir / f"kaldi_dict.{in_labels}.txt"
    if not in_units_file.exists():

        logger.info(f"Creating {in_units_file}")

        with open(in_units_file, "w") as f:
            print("<eps> 0", file=f)
            i = 1
            for symb in vocab.symbols[vocab.nspecial :]:
                if not symb.startswith("madeupword"):
                    print(f"{symb} {i}", file=f)
                    i += 1
    return in_units_file


def create_lexicon(
    cfg: KaldiInitializerConfig,
    fst_dir: Path,
    unique_label: str,
    in_units_file: Path,
    out_words_file: Path,
) -> (Path, Path):

    disambig_in_units_file = fst_dir / f"kaldi_dict.{cfg.in_labels}_disambig.txt"
    lexicon_file = fst_dir / f"kaldi_lexicon.{unique_label}.txt"
    disambig_lexicon_file = fst_dir / f"kaldi_lexicon.{unique_label}_disambig.txt"
    if (
        not lexicon_file.exists()
        or not disambig_lexicon_file.exists()
        or not disambig_in_units_file.exists()
    ):
        logger.info(f"Creating {lexicon_file} (in units file: {in_units_file})")

        assert cfg.wav2letter_lexicon is not None or cfg.in_labels == cfg.out_labels

        if cfg.wav2letter_lexicon is not None:
            lm_words = set()
            with open(out_words_file, "r") as lm_dict_f:
                for line in lm_dict_f:
                    lm_words.add(line.split()[0])

            num_skipped = 0
            total = 0
            with open(cfg.wav2letter_lexicon, "r") as w2l_lex_f, open(
                lexicon_file, "w"
            ) as out_f:
                for line in w2l_lex_f:
                    items = line.rstrip().split("\t")
                    assert len(items) == 2, items
                    if items[0] in lm_words:
                        print(items[0], items[1], file=out_f)
                    else:
                        num_skipped += 1
                        logger.debug(
                            f"Skipping word {items[0]} as it was not found in LM"
                        )
                    total += 1
            if num_skipped > 0:
                logger.warning(
                    f"Skipped {num_skipped} out of {total} words as they were not found in LM"
                )
        else:
            with open(in_units_file, "r") as in_f, open(lexicon_file, "w") as out_f:
                for line in in_f:
                    symb = line.split()[0]
                    if symb != "<eps>" and symb != "<ctc_blank>" and symb != "<SIL>":
                        print(symb, symb, file=out_f)

        lex_disambig_path = (
            Path(cfg.kaldi_root) / "egs/wsj/s5/utils/add_lex_disambig.pl"
        )
        res = subprocess.run(
            [lex_disambig_path, lexicon_file, disambig_lexicon_file],
            check=True,
            capture_output=True,
        )
        ndisambig = int(res.stdout)
        disamib_path = Path(cfg.kaldi_root) / "egs/wsj/s5/utils/add_disambig.pl"
        res = subprocess.run(
            [disamib_path, "--include-zero", in_units_file, str(ndisambig)],
            check=True,
            capture_output=True,
        )
        with open(disambig_in_units_file, "wb") as f:
            f.write(res.stdout)

    return disambig_lexicon_file, disambig_in_units_file


def create_G(
    kaldi_root: Path, fst_dir: Path, lm_arpa: Path, arpa_base: str
) -> (Path, Path):

    out_words_file = fst_dir / f"kaldi_dict.{arpa_base}.txt"
    grammar_graph = fst_dir / f"G_{arpa_base}.fst"
    if not grammar_graph.exists() or not out_words_file.exists():
        logger.info(f"Creating {grammar_graph}")
        arpa2fst = kaldi_root / "src/lmbin/arpa2fst"
        subprocess.run(
            [
                arpa2fst,
                "--disambig-symbol=#0",
                f"--write-symbol-table={out_words_file}",
                lm_arpa,
                grammar_graph,
            ],
            check=True,
        )
    return grammar_graph, out_words_file


def create_L(
    kaldi_root: Path,
    fst_dir: Path,
    unique_label: str,
    lexicon_file: Path,
    in_units_file: Path,
    out_words_file: Path,
) -> Path:
    lexicon_graph = fst_dir / f"L.{unique_label}.fst"

    if not lexicon_graph.exists():
        logger.info(f"Creating {lexicon_graph} (in units: {in_units_file})")
        make_lex = kaldi_root / "egs/wsj/s5/utils/make_lexicon_fst.pl"
        fstcompile = kaldi_root / "tools/openfst-1.6.7/bin/fstcompile"
        fstaddselfloops = kaldi_root / "src/fstbin/fstaddselfloops"
        fstarcsort = kaldi_root / "tools/openfst-1.6.7/bin/fstarcsort"

        def write_disambig_symbol(file):
            with open(file, "r") as f:
                for line in f:
                    items = line.rstrip().split()
                    if items[0] == "#0":
                        out_path = str(file) + "_disamig"
                        with open(out_path, "w") as out_f:
                            print(items[1], file=out_f)
                            return out_path

            return None

        in_disambig_sym = write_disambig_symbol(in_units_file)
        assert in_disambig_sym is not None
        out_disambig_sym = write_disambig_symbol(out_words_file)
        assert out_disambig_sym is not None

        try:
            with open(lexicon_graph, "wb") as out_f:
                res = subprocess.run(
                    [make_lex, lexicon_file], capture_output=True, check=True
                )
                assert len(res.stderr) == 0, res.stderr.decode("utf-8")
                res = subprocess.run(
                    [
                        fstcompile,
                        f"--isymbols={in_units_file}",
                        f"--osymbols={out_words_file}",
                        "--keep_isymbols=false",
                        "--keep_osymbols=false",
                    ],
                    input=res.stdout,
                    capture_output=True,
                )
                assert len(res.stderr) == 0, res.stderr.decode("utf-8")
                res = subprocess.run(
                    [fstaddselfloops, in_disambig_sym, out_disambig_sym],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstarcsort, "--sort_type=olabel"],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                out_f.write(res.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            os.remove(lexicon_graph)
            raise
        except AssertionError:
            os.remove(lexicon_graph)
            raise

    return lexicon_graph


def create_LG(
    kaldi_root: Path,
    fst_dir: Path,
    unique_label: str,
    lexicon_graph: Path,
    grammar_graph: Path,
) -> Path:
    lg_graph = fst_dir / f"LG.{unique_label}.fst"

    if not lg_graph.exists():
        logger.info(f"Creating {lg_graph}")

        fsttablecompose = kaldi_root / "src/fstbin/fsttablecompose"
        fstdeterminizestar = kaldi_root / "src/fstbin/fstdeterminizestar"
        fstminimizeencoded = kaldi_root / "src/fstbin/fstminimizeencoded"
        fstpushspecial = kaldi_root / "src/fstbin/fstpushspecial"
        fstarcsort = kaldi_root / "tools/openfst-1.6.7/bin/fstarcsort"

        try:
            with open(lg_graph, "wb") as out_f:
                res = subprocess.run(
                    [fsttablecompose, lexicon_graph, grammar_graph],
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [
                        fstdeterminizestar,
                        "--use-log=true",
                    ],
                    input=res.stdout,
                    capture_output=True,
                )
                res = subprocess.run(
                    [fstminimizeencoded],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstpushspecial],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstarcsort, "--sort_type=ilabel"],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                out_f.write(res.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            os.remove(lg_graph)
            raise

    return lg_graph


def create_H(
    kaldi_root: Path,
    fst_dir: Path,
    disambig_out_units_file: Path,
    in_labels: str,
    vocab: Dictionary,
    blk_sym: str,
    silence_symbol: Optional[str],
) -> (Path, Path, Path):
    h_graph = (
        fst_dir / f"H.{in_labels}{'_' + silence_symbol if silence_symbol else ''}.fst"
    )
    h_out_units_file = fst_dir / f"kaldi_dict.h_out.{in_labels}.txt"
    disambig_in_units_file_int = Path(str(h_graph) + "isym_disambig.int")
    disambig_out_units_file_int = Path(str(disambig_out_units_file) + ".int")
    if (
        not h_graph.exists()
        or not h_out_units_file.exists()
        or not disambig_in_units_file_int.exists()
    ):
        logger.info(f"Creating {h_graph}")
        eps_sym = "<eps>"

        num_disambig = 0
        osymbols = []

        with open(disambig_out_units_file, "r") as f, open(
            disambig_out_units_file_int, "w"
        ) as out_f:
            for line in f:
                symb, id = line.rstrip().split()
                if line.startswith("#"):
                    num_disambig += 1
                    print(id, file=out_f)
                else:
                    if len(osymbols) == 0:
                        assert symb == eps_sym, symb
                    osymbols.append((symb, id))

        i_idx = 0
        isymbols = [(eps_sym, 0)]

        imap = {}

        for i, s in enumerate(vocab.symbols):
            i_idx += 1
            isymbols.append((s, i_idx))
            imap[s] = i_idx

        fst_str = []

        node_idx = 0
        root_node = node_idx

        special_symbols = [blk_sym]
        if silence_symbol is not None:
            special_symbols.append(silence_symbol)

        for ss in special_symbols:
            fst_str.append("{} {} {} {}".format(root_node, root_node, ss, eps_sym))

        for symbol, _ in osymbols:
            if symbol == eps_sym or symbol.startswith("#"):
                continue

            node_idx += 1
            # 1. from root to emitting state
            fst_str.append("{} {} {} {}".format(root_node, node_idx, symbol, symbol))
            # 2. from emitting state back to root
            fst_str.append("{} {} {} {}".format(node_idx, root_node, eps_sym, eps_sym))
            # 3. from emitting state to optional blank state
            pre_node = node_idx
            node_idx += 1
            for ss in special_symbols:
                fst_str.append("{} {} {} {}".format(pre_node, node_idx, ss, eps_sym))
            # 4. from blank state back to root
            fst_str.append("{} {} {} {}".format(node_idx, root_node, eps_sym, eps_sym))

        fst_str.append("{}".format(root_node))

        fst_str = "\n".join(fst_str)
        h_str = str(h_graph)
        isym_file = h_str + ".isym"

        with open(isym_file, "w") as f:
            for sym, id in isymbols:
                f.write("{} {}\n".format(sym, id))

        with open(h_out_units_file, "w") as f:
            for sym, id in osymbols:
                f.write("{} {}\n".format(sym, id))

        with open(disambig_in_units_file_int, "w") as f:
            disam_sym_id = len(isymbols)
            for _ in range(num_disambig):
                f.write("{}\n".format(disam_sym_id))
                disam_sym_id += 1

        fstcompile = kaldi_root / "tools/openfst-1.6.7/bin/fstcompile"
        fstaddselfloops = kaldi_root / "src/fstbin/fstaddselfloops"
        fstarcsort = kaldi_root / "tools/openfst-1.6.7/bin/fstarcsort"

        try:
            with open(h_graph, "wb") as out_f:
                res = subprocess.run(
                    [
                        fstcompile,
                        f"--isymbols={isym_file}",
                        f"--osymbols={h_out_units_file}",
                        "--keep_isymbols=false",
                        "--keep_osymbols=false",
                    ],
                    input=str.encode(fst_str),
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [
                        fstaddselfloops,
                        disambig_in_units_file_int,
                        disambig_out_units_file_int,
                    ],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstarcsort, "--sort_type=olabel"],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                out_f.write(res.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            os.remove(h_graph)
            raise
    return h_graph, h_out_units_file, disambig_in_units_file_int


def create_HLGa(
    kaldi_root: Path,
    fst_dir: Path,
    unique_label: str,
    h_graph: Path,
    lg_graph: Path,
    disambig_in_words_file_int: Path,
) -> Path:
    hlga_graph = fst_dir / f"HLGa.{unique_label}.fst"

    if not hlga_graph.exists():
        logger.info(f"Creating {hlga_graph}")

        fsttablecompose = kaldi_root / "src/fstbin/fsttablecompose"
        fstdeterminizestar = kaldi_root / "src/fstbin/fstdeterminizestar"
        fstrmsymbols = kaldi_root / "src/fstbin/fstrmsymbols"
        fstrmepslocal = kaldi_root / "src/fstbin/fstrmepslocal"
        fstminimizeencoded = kaldi_root / "src/fstbin/fstminimizeencoded"

        try:
            with open(hlga_graph, "wb") as out_f:
                res = subprocess.run(
                    [
                        fsttablecompose,
                        h_graph,
                        lg_graph,
                    ],
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstdeterminizestar, "--use-log=true"],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstrmsymbols, disambig_in_words_file_int],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstrmepslocal],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstminimizeencoded],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                out_f.write(res.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            os.remove(hlga_graph)
            raise

    return hlga_graph


def create_HLa(
    kaldi_root: Path,
    fst_dir: Path,
    unique_label: str,
    h_graph: Path,
    l_graph: Path,
    disambig_in_words_file_int: Path,
) -> Path:
    hla_graph = fst_dir / f"HLa.{unique_label}.fst"

    if not hla_graph.exists():
        logger.info(f"Creating {hla_graph}")

        fsttablecompose = kaldi_root / "src/fstbin/fsttablecompose"
        fstdeterminizestar = kaldi_root / "src/fstbin/fstdeterminizestar"
        fstrmsymbols = kaldi_root / "src/fstbin/fstrmsymbols"
        fstrmepslocal = kaldi_root / "src/fstbin/fstrmepslocal"
        fstminimizeencoded = kaldi_root / "src/fstbin/fstminimizeencoded"

        try:
            with open(hla_graph, "wb") as out_f:
                res = subprocess.run(
                    [
                        fsttablecompose,
                        h_graph,
                        l_graph,
                    ],
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstdeterminizestar, "--use-log=true"],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstrmsymbols, disambig_in_words_file_int],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstrmepslocal],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                res = subprocess.run(
                    [fstminimizeencoded],
                    input=res.stdout,
                    capture_output=True,
                    check=True,
                )
                out_f.write(res.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            os.remove(hla_graph)
            raise

    return hla_graph


def create_HLG(
    kaldi_root: Path,
    fst_dir: Path,
    unique_label: str,
    hlga_graph: Path,
    prefix: str = "HLG",
) -> Path:
    hlg_graph = fst_dir / f"{prefix}.{unique_label}.fst"

    if not hlg_graph.exists():
        logger.info(f"Creating {hlg_graph}")

        add_self_loop = script_dir / "add-self-loop-simple"
        kaldi_src = kaldi_root / "src"
        kaldi_lib = kaldi_src / "lib"

        try:
            if not add_self_loop.exists():
                fst_include = kaldi_root / "tools/openfst-1.6.7/include"
                add_self_loop_src = script_dir / "add-self-loop-simple.cc"

                subprocess.run(
                    [
                        "c++",
                        f"-I{kaldi_src}",
                        f"-I{fst_include}",
                        f"-L{kaldi_lib}",
                        add_self_loop_src,
                        "-lkaldi-base",
                        "-lkaldi-fstext",
                        "-o",
                        add_self_loop,
                    ],
                    check=True,
                )

            my_env = os.environ.copy()
            my_env["LD_LIBRARY_PATH"] = f"{kaldi_lib}:{my_env['LD_LIBRARY_PATH']}"

            subprocess.run(
                [
                    add_self_loop,
                    hlga_graph,
                    hlg_graph,
                ],
                check=True,
                capture_output=True,
                env=my_env,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"cmd: {e.cmd}, err: {e.stderr.decode('utf-8')}")
            raise

    return hlg_graph


def initalize_kaldi(cfg: KaldiInitializerConfig) -> Path:
    if cfg.fst_dir is None:
        cfg.fst_dir = osp.join(cfg.data_dir, "kaldi")
    if cfg.out_labels is None:
        cfg.out_labels = cfg.in_labels

    kaldi_root = Path(cfg.kaldi_root)
    data_dir = Path(cfg.data_dir)
    fst_dir = Path(cfg.fst_dir)
    fst_dir.mkdir(parents=True, exist_ok=True)

    arpa_base = osp.splitext(osp.basename(cfg.lm_arpa))[0]
    unique_label = f"{cfg.in_labels}.{arpa_base}"

    with open(data_dir / f"dict.{cfg.in_labels}.txt", "r") as f:
        vocab = Dictionary.load(f)

    in_units_file = create_units(fst_dir, cfg.in_labels, vocab)

    grammar_graph, out_words_file = create_G(
        kaldi_root, fst_dir, Path(cfg.lm_arpa), arpa_base
    )

    disambig_lexicon_file, disambig_L_in_units_file = create_lexicon(
        cfg, fst_dir, unique_label, in_units_file, out_words_file
    )

    h_graph, h_out_units_file, disambig_in_units_file_int = create_H(
        kaldi_root,
        fst_dir,
        disambig_L_in_units_file,
        cfg.in_labels,
        vocab,
        cfg.blank_symbol,
        cfg.silence_symbol,
    )
    lexicon_graph = create_L(
        kaldi_root,
        fst_dir,
        unique_label,
        disambig_lexicon_file,
        disambig_L_in_units_file,
        out_words_file,
    )
    lg_graph = create_LG(
        kaldi_root, fst_dir, unique_label, lexicon_graph, grammar_graph
    )
    hlga_graph = create_HLGa(
        kaldi_root, fst_dir, unique_label, h_graph, lg_graph, disambig_in_units_file_int
    )
    hlg_graph = create_HLG(kaldi_root, fst_dir, unique_label, hlga_graph)

    # for debugging
    # hla_graph = create_HLa(kaldi_root, fst_dir, unique_label, h_graph, lexicon_graph, disambig_in_units_file_int)
    # hl_graph = create_HLG(kaldi_root, fst_dir, unique_label, hla_graph, prefix="HL_looped")
    # create_HLG(kaldi_root, fst_dir, "phnc", h_graph, prefix="H_looped")

    return hlg_graph


@hydra.main(config_path=config_path, config_name="kaldi_initializer")
def cli_main(cfg: KaldiInitializerConfig) -> None:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)
    initalize_kaldi(cfg)


if __name__ == "__main__":

    logging.root.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)

    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "kaldi_initializer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "kaldi_initializer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=KaldiInitializerConfig)

    cli_main()
