import io
import os
import logging
from fairseq import checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import (
    read_from_stored_zip,
    is_sf_audio_data,
    parse_path,
)
from fairseq.models.streaming.modules.monotonic_transformer_decoder import (
    TransformerMonotonicDecoder,
)
from fairseq.models.streaming.modules.fixed_pre_decision import (
    WaitKAttentionFixedStride,
)
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig
from typing import Optional
from omegaconf import OmegaConf

try:
    from simuleval.agents import GenericAgent

    IS_SIMULEVAL_IMPORTED = True
except Exception:
    GenericAgent = object
    IS_SIMULEVAL_IMPORTED = False


def test_time_waitk_agent(agent: GenericAgent):
    class TestTimeWaitKAgent(agent):
        @staticmethod
        def add_args(parser):
            super(agent, TestTimeWaitKAgent).add_args(parser)
            try:
                # Try to add new add_args
                WaitKAttentionFixedStride.add_args(parser)
            except:
                pass

        def __init__(self, decoder, args) -> None:
            assert IS_SIMULEVAL_IMPORTED
            args.simul_type = "waitk_fixed_pre_decision"
            decoder = TransformerMonotonicDecoder.from_offline_decoder(decoder, args)
            super().__init__(decoder, args)

    TestTimeWaitKAgent.__name__ = agent.__name__

    return TestTimeWaitKAgent


def load_fairseq_model(args):
    filename = args.checkpoint
    config_yaml = args.config_yaml
    w2v_yaml = getattr(args, "wav2vec_yaml", None)

    logger = logging.getLogger("fairseq.models.wav2vec.wav2vec2_asr")
    logger.disabled = True
    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))

    state = checkpoint_utils.load_checkpoint_to_cpu(filename)

    if config_yaml is not None:
        state["cfg"]["task"].data = os.path.dirname(config_yaml)  
        state["cfg"]["task"].config = os.path.basename(config_yaml)  

    if  w2v_yaml is not None:
        w2v_args = OmegaConf.create()
        w2v_args.model = OmegaConf.load(w2v_yaml)
        w2v_args.task = AudioPretrainingConfig(_name="audio_pretraining", data=state["cfg"]["task"].data)
        state["cfg"].model.w2v_args = w2v_args



    task = tasks.setup_task(state["cfg"]["task"])

    state["cfg"]["model"].max_positions = 1024
    state["cfg"]["model"].max_source_positions = 1024
    state["cfg"]["model"].max_target_positions = 1024
    state["cfg"]["model"].load_pretrained_decoder_from = None
    model = task.build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    model.share_memory()
    return model


def get_audio_file_path(path_of_fp):
    _path, slice_ptr = parse_path(path_of_fp)
    if len(slice_ptr) == 2:
        byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
        assert is_sf_audio_data(byte_data)
        path_of_fp = io.BytesIO(byte_data)
    return path_of_fp
