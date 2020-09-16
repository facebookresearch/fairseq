from fairseq import checkpoint_utils
from .modules.sequence_encoder import SequenceEncoder

from .modules.augmented_memory_attention import (
    AugmentedMemoryConvTransformerEncoder
)
from .convtransformer import (
    ConvTransformerModel,
    convtransformer_espnet
)

from fairseq.models import (
    register_model,
    register_model_architecture,
)

from examples.simultaneous_translation.models import (
    TransformerMonotonicDecoder
)


def enable_streaming(klass):
    class StreamSeq2SeqModel(klass):

        @staticmethod
        def add_args(parser):
            super(StreamSeq2SeqModel, StreamSeq2SeqModel).add_args(parser)
            parser.add_argument("--segment-size", type=int, required=True,
                                help="Length of the segment.")
            parser.add_argument("--left-context", type=int, default=0,
                                help="Left context for the segment.")
            parser.add_argument("--right-context", type=int, default=0,
                                help="Right context for the segment.")
            parser.add_argument("--max-memory-size", type=int, default=-1,
                                help="Right context for the segment.")

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel


@register_model("convtransformer_augmented_memory")
@enable_streaming
class AugmentedMemoryConvTransformerModel(ConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = SequenceEncoder(
            args, AugmentedMemoryConvTransformerEncoder(args)
        )

        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )

        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return TransformerMonotonicDecoder(
            args, task.target_dictionary, embed_tokens)


@register_model_architecture(
    "convtransformer_augmented_memory",
    "convtransformer_augmented_memory"
)
def stream_convtransformer_espnet(args):
    convtransformer_espnet(args)
