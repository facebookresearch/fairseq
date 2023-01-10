from .offline_w2v_encoder import OfflineWav2VecEncoderAgent
from .online_text_decoder import TestTimeWaitKOnlineSpeechToTextDecoderAgent
from .spm_detokenizer import SentencePieceModelDetokenizerAgent
from .fairseq_pipeline import FairseqAgentPipeline
from fairseq.models.streaming.utils.common import load_fairseq_model


class TestTimeWaitKS2T(FairseqAgentPipeline):
    pipeline = [
        OfflineWav2VecEncoderAgent,
        TestTimeWaitKOnlineSpeechToTextDecoderAgent,
        SentencePieceModelDetokenizerAgent,
    ]

    def __init__(self, args) -> None:

        fairseq_model = load_fairseq_model(args)

        encoder = self.pipeline[0](fairseq_model.encoder, args)
        decoder = self.pipeline[1](fairseq_model.decoder, args)
        detokenizer = self.pipeline[2](args)

        module_list = [encoder, decoder, detokenizer]

        super().__init__(module_list)
