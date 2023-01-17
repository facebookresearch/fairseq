from simuleval.utils import entrypoint
from fairseq.models.streaming.agents import (
    FairseqAgentPipeline,
    OfflineWav2VecEncoderAgent,
    TestTimeWaitKOnlineSpeechToTextDecoderAgent,
    SimpleTextFilterAgent,
)
from fairseq.models.streaming.utils.common import load_fairseq_model


@entrypoint
class TestTimeWaitKS2T(FairseqAgentPipeline):
    pipeline = [
        OfflineWav2VecEncoderAgent,
        TestTimeWaitKOnlineSpeechToTextDecoderAgent,
        SimpleTextFilterAgent,
    ]

    def __init__(self, args) -> None:

        fairseq_model = load_fairseq_model(args)

        encoder = self.pipeline[0](fairseq_model.encoder, args)
        decoder = self.pipeline[1](fairseq_model.decoder, args)
        filter = self.pipeline[2](args)
        module_list = [encoder, decoder, filter]

        super().__init__(module_list)
