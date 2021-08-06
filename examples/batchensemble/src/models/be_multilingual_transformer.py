from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import (
    MultilingualTransformerModel, multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import TransformerEncoder

from .be_transformer import BatchEnsembleTransformerDecoder


@register_model("be_multilingual_transformer")
class BatchEnsembleMultilingualTransformer(MultilingualTransformerModel):
    """A variant of standard multilingual Transformer models whose decoder
    supports uses BatchEnsemble.
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        MultilingualTransformerModel.add_args(parser)

    @classmethod
    def _get_module_class(
        cls, is_encoder, args, lang_dict, embed_tokens, langs
    ):
        module_class = (
            TransformerEncoder
            if is_encoder
            else BatchEnsembleTransformerDecoder
        )

        return module_class(args, lang_dict, embed_tokens)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_multilingual_transformer"
)
def batch_ensemble_multilingual_architecture(args):
    multilingual_transformer_iwslt_de_en(args)


@register_model_architecture(
    "be_multilingual_transformer",
    "batch_ensemble_phat_multilingual_transformer"
)
def batch_ensemble_phat_multilingual_architecture(args):
    # Latent Depth number of layers
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 24)

    batch_ensemble_multilingual_architecture(args)
