#!/usr/bin/env python3

# import models/encoder/decoder to be tested
from examples.speech_recognition.models.vggtransformer import (
    TransformerDecoder,
    VGGTransformerEncoder,
    VGGTransformerModel,
    vggtransformer_1,
    vggtransformer_2,
    vggtransformer_base,
)

# import base test class
from .asr_test_base import (
    DEFAULT_TEST_VOCAB_SIZE,
    TestFairseqDecoderBase,
    TestFairseqEncoderBase,
    TestFairseqEncoderDecoderModelBase,
    get_dummy_dictionary,
    get_dummy_encoder_output,
    get_dummy_input,
)


class VGGTransformerModelTest_mid(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_1 use 14 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_1, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerModelTest_big(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_2 use 16 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((1024, 16, 4096, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_2, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerModelTest_base(TestFairseqEncoderDecoderModelBase):
    def setUp(self):
        def override_config(args):
            """
            vggtrasformer_base use 12 layers of transformer,
            for testing purpose, it is too expensive. For fast turn-around
            test, reduce the number of layers to 3.
            """
            args.transformer_enc_config = (
                "((512, 8, 2048, True, 0.15, 0.15, 0.15),) * 3"
            )

        super().setUp()
        extra_args_setter = [vggtransformer_base, override_config]

        self.setUpModel(VGGTransformerModel, extra_args_setter)
        self.setUpInput(get_dummy_input(T=50, D=80, B=5, K=DEFAULT_TEST_VOCAB_SIZE))


class VGGTransformerEncoderTest(TestFairseqEncoderBase):
    def setUp(self):
        super().setUp()

        self.setUpInput(get_dummy_input(T=50, D=80, B=5))

    def test_forward(self):
        print("1. test standard vggtransformer")
        self.setUpEncoder(VGGTransformerEncoder(input_feat_per_channel=80))
        super().test_forward()
        print("2. test vggtransformer with limited right context")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80, transformer_context=(-1, 5)
            )
        )
        super().test_forward()
        print("3. test vggtransformer with limited left context")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80, transformer_context=(5, -1)
            )
        )
        super().test_forward()
        print("4. test vggtransformer with limited right context and sampling")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80,
                transformer_context=(-1, 12),
                transformer_sampling=(2, 2),
            )
        )
        super().test_forward()
        print("5. test vggtransformer with windowed context and sampling")
        self.setUpEncoder(
            VGGTransformerEncoder(
                input_feat_per_channel=80,
                transformer_context=(12, 12),
                transformer_sampling=(2, 2),
            )
        )


class TransformerDecoderTest(TestFairseqDecoderBase):
    def setUp(self):
        super().setUp()

        dict = get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE)
        decoder = TransformerDecoder(dict)
        dummy_encoder_output = get_dummy_encoder_output(encoder_out_shape=(50, 5, 256))

        self.setUpDecoder(decoder)
        self.setUpInput(dummy_encoder_output)
        self.setUpPrevOutputTokens()
