# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import unittest
from io import StringIO

import torch
from fairseq import fb_hub


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestTranslationHub(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @torch.no_grad()
    def test_transformer_wmt14_en_fr(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load an En-Fr Transformer model trained on WMT'14 data
            en2fr = fb_hub.load(
                "transformer.wmt14.en-fr", tokenizer="moses", bpe="subword_nmt"
            )
            en2fr.eval()  # disable dropout

            # Translate with beam search
            fr = en2fr.translate("Hello world!", beam=5)
            self.assertEqual(fr, "Bonjour à tous !")

            # Manually tokenize
            en_toks = en2fr.tokenize("Hello world!")
            self.assertEqual(en_toks, "Hello world !")

            # Manually apply BPE
            en_bpe = en2fr.apply_bpe(en_toks)
            self.assertEqual(en_bpe, "H@@ ello world !")

            # Manually binarize
            en_bin = en2fr.binarize(en_bpe)
            self.assertEqual(en_bin.tolist(), [329, 14044, 682, 812, 2])

            # Generate five translations with top-k sampling
            fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
            self.assertEqual(len(fr_bin), 5)

            # Convert one of the samples to a string and detokenize
            fr_sample = fr_bin[0]["tokens"]
            fr_bpe = en2fr.string(fr_sample)
            fr_toks = en2fr.remove_bpe(fr_bpe)
            fr = en2fr.detokenize(fr_toks)
            self.assertEqual(fr, en2fr.decode(fr_sample))

            # Batched translation
            fr_batch = en2fr.translate(["Hello world", "The cat sat on the mat."])
            self.assertEqual(
                fr_batch, ["Bonjour à tous.", "Le chat était assis sur le tapis."]
            )

    @torch.no_grad()
    def test_transformer_wmt19_en_de_single_model(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load an En-De Transformer model trained on WMT'19 data
            en2de = fb_hub.load(
                "transformer.wmt19.en-de.single_model", tokenizer="moses", bpe="fastbpe"
            )
            en2de.eval()  # disable dropout

            # Access the underlying TransformerModel
            self.assertTrue(isinstance(en2de.models[0], torch.nn.Module))

            # Translate from En-De
            de = en2de.translate(
                "PyTorch Hub is a pre-trained model repository designed to facilitate research reproducibility."
            )
            self.assertEqual(
                de,
                "PyTorch Hub ist ein vorgefertigtes Modell-Repository, das die Reproduzierbarkeit der Forschung erleichtern soll.",
            )


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestLMHub(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @torch.no_grad()
    def test_transformer_lm_wmt19_en(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load an English LM trained on WMT'19 News Crawl data
            en_lm = fb_hub.load("transformer_lm.wmt19.en")
            en_lm.eval()  # disable dropout

            # Sample from the language model
            en_lm.sample(
                "Barack Obama", beam=1, sampling=True, sampling_topk=10, temperature=0.8
            )

            ppl = (
                en_lm.score("Barack Obama is coming to Sydney and New Zealand")[
                    "positional_scores"
                ]
                .mean()
                .neg()
                .exp()
            )
            self.assertAlmostEqual(ppl.item(), 4.2739, places=4)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestRobertaHub(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @torch.no_grad()
    def test_roberta_base(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load RoBERTa
            roberta = fb_hub.load("roberta.base")
            roberta.eval()  # disable dropout

            # Apply Byte-Pair Encoding (BPE) to input text
            tokens = roberta.encode("Hello world!")
            self.assertEqual(tokens.tolist(), [0, 31414, 232, 328, 2])
            self.assertEqual(roberta.decode(tokens), "Hello world!")

            # Extract the last layer's features
            last_layer_features = roberta.extract_features(tokens)
            self.assertEqual(last_layer_features.size(), torch.Size([1, 5, 768]))

            # Extract all layer's features (layer 0 is the embedding layer)
            all_layers = roberta.extract_features(tokens, return_all_hiddens=True)
            self.assertEqual(len(all_layers), 13)
            self.assertTrue(torch.all(all_layers[-1] == last_layer_features))

            # Register a new (randomly initialized) classification head
            roberta.register_classification_head("new_task", num_classes=3)
            logprobs = roberta.predict("new_task", tokens)  # noqa

            # Test mask filling
            res = roberta.fill_mask(
                "The first Star wars movie came out in <mask>", topk=3
            )
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0][2], " 1977")

    @torch.no_grad()
    def test_roberta_large_mnli(self):
        with contextlib.redirect_stdout(StringIO()):
            # Download RoBERTa already finetuned for MNLI
            roberta = fb_hub.load("roberta.large.mnli")
            roberta.eval()  # disable dropout for evaluation

            # Encode a pair of sentences and make a prediction
            tokens = roberta.encode(
                "Roberta is a heavily optimized version of BERT.",
                "Roberta is not very optimized.",
            )
            prediction = roberta.predict("mnli", tokens).argmax().item()
            self.assertEqual(prediction, 0)  # contradiction

            # Encode another pair of sentences
            tokens = roberta.encode(
                "Roberta is a heavily optimized version of BERT.",
                "Roberta is based on BERT.",
            )
            prediction = roberta.predict("mnli", tokens).argmax().item()
            self.assertEqual(prediction, 2)  # entailment

            # Test batched prediction
            from fairseq.data.data_utils import collate_tokens

            batch_of_pairs = [
                [
                    "Roberta is a heavily optimized version of BERT.",
                    "Roberta is not very optimized.",
                ],
                [
                    "Roberta is a heavily optimized version of BERT.",
                    "Roberta is based on BERT.",
                ],
                ["potatoes are awesome.", "I like to run."],
                ["Mars is very far from earth.", "Mars is very close."],
            ]
            batch = collate_tokens(
                [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
            )
            logprobs = roberta.predict("mnli", batch)
            self.assertEqual(logprobs.argmax(dim=1).tolist(), [0, 2, 1, 0])

    @torch.no_grad()
    def test_roberta_large_wsc(self):
        with contextlib.redirect_stdout(StringIO()):
            roberta = fb_hub.load("roberta.large.wsc", user_dir="examples/roberta/wsc")
            roberta.eval()  # disable dropout

            ans = roberta.disambiguate_pronoun(
                "The _trophy_ would not fit in the brown suitcase because [it] was too big."
            )
            self.assertTrue(ans)

            ans = roberta.disambiguate_pronoun(
                "The trophy would not fit in the brown _suitcase_ because [it] was too big."
            )
            self.assertFalse(ans)

            ans = roberta.disambiguate_pronoun(
                "The city councilmen refused the demonstrators a permit because [they] feared violence."
            )
            self.assertEqual(ans, "The city councilmen")

            ans = roberta.disambiguate_pronoun(
                "The city councilmen refused the demonstrators a permit because [they] advocated violence."
            )
            self.assertEqual(ans, "demonstrators")

    @torch.no_grad()
    def test_camembert(self):
        with contextlib.redirect_stdout(StringIO()):
            camembert = fb_hub.load("camembert.v0")
            camembert.eval()  # disable dropout

            # Filling masks
            masked_line = "Le camembert est <mask> :)"
            res = camembert.fill_mask(masked_line, topk=3)
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0][2], " délicieux")

            # Extract the last layer's features
            line = "J'aime le camembert!"
            tokens = camembert.encode(line)
            last_layer_features = camembert.extract_features(tokens)
            self.assertEqual(last_layer_features.size(), torch.Size([1, 10, 768]))

            # Extract all layer's features (layer 0 is the embedding layer)
            all_layers = camembert.extract_features(tokens, return_all_hiddens=True)
            self.assertEqual(len(all_layers), 13)
            self.assertTrue(torch.all(all_layers[-1] == last_layer_features))

    @torch.no_grad()
    def test_xlmr(self):
        with contextlib.redirect_stdout(StringIO()):
            xlmr = fb_hub.load("xlmr.large")
            xlmr.eval()  # disable dropout

            # Test sentencepiece
            en_tokens = xlmr.encode("Hello world!")
            self.assertEqual(en_tokens.tolist(), [0, 35378, 8999, 38, 2])
            xlmr.decode(en_tokens)  # 'Hello world!'

            zh_tokens = xlmr.encode("你好，世界")
            self.assertEqual(zh_tokens.tolist(), [0, 6, 124084, 4, 3221, 2])
            xlmr.decode(zh_tokens)  # '你好，世界'

            hi_tokens = xlmr.encode("नमस्ते दुनिया")
            self.assertEqual(hi_tokens.tolist(), [0, 68700, 97883, 29405, 2])
            xlmr.decode(hi_tokens)  # 'नमस्ते दुनिया'

            ar_tokens = xlmr.encode("مرحبا بالعالم")
            self.assertEqual(ar_tokens.tolist(), [0, 665, 193478, 258, 1705, 77796, 2])
            xlmr.decode(ar_tokens)  # 'مرحبا بالعالم'

            fr_tokens = xlmr.encode("Bonjour le monde")
            self.assertEqual(fr_tokens.tolist(), [0, 84602, 95, 11146, 2])
            xlmr.decode(fr_tokens)  # 'Bonjour le monde'

            # Extract the last layer's features
            last_layer_features = xlmr.extract_features(zh_tokens)
            self.assertEqual(last_layer_features.size(), torch.Size([1, 6, 1024]))

            # Extract all layer's features (layer 0 is the embedding layer)
            all_layers = xlmr.extract_features(zh_tokens, return_all_hiddens=True)
            self.assertEqual(len(all_layers), 25)
            self.assertTrue(torch.all(all_layers[-1] == last_layer_features))


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestBartHub(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @torch.no_grad()
    def test_bart_base(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load BART
            bart = fb_hub.load("bart.base")
            bart.eval()  # disable dropout

            # Test mask filling (beam = topk = 3)
            res = bart.fill_mask(["The cat <mask> on the <mask>."], topk=3)[0]
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0][0], "The cat was on the ground.")
            self.assertEqual(res[1][0], "The cat was on the floor.")
            self.assertEqual(res[2][0], "The cat was sitting on the couch")

            # Test mask filling (beam = 10, topk = 3)
            res = bart.fill_mask(["The cat <mask> on the <mask>."], topk=3, beam=10)[0]
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0][0], "The cat was on the ground.")
            self.assertEqual(res[1][0], "The cat was on the floor.")
            self.assertEqual(res[2][0], "The cat sleeps on the couch.")

            # Test mask filling (beam = 10, topk = 3, match_source_len = False)
            res = bart.fill_mask(
                ["The cat <mask> on the <mask>."],
                topk=3,
                beam=10,
                match_source_len=False,
            )[0]
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0][0], "The cat was on the ground.")
            self.assertEqual(res[1][0], "The cat was asleep on the couch.")
            self.assertEqual(res[2][0], "The cat was on the floor.")

            # Test mask filling (beam = 10, topk = 3) and batch size > 1
            res = bart.fill_mask(["The cat <mask> on the <mask>.", "The dog <mask> on the <mask>."],
                                 topk=3, beam=10)
            self.assertEqual(len(res), 2)
            self.assertEqual(len(res[0]), 3)
            self.assertEqual(res[0][0][0], "The cat was on the ground.")
            self.assertEqual(res[0][1][0], "The cat was on the floor.")
            self.assertEqual(res[0][2][0], "The cat sleeps on the couch.")
            self.assertEqual(len(res[1]), 3)
            self.assertEqual(res[1][0][0], "The dog was on the ground.")
            self.assertEqual(res[1][1][0], "The dog lay on the ground.")
            self.assertEqual(res[1][2][0], "The dog was asleep on the couch")

    @torch.no_grad()
    def test_bart_large(self):
        with contextlib.redirect_stdout(StringIO()):
            # Load BART
            bart = fb_hub.load("bart.large")
            bart.eval()  # disable dropout

            # Apply Byte-Pair Encoding (BPE) to input text
            tokens = bart.encode("Hello world!")
            self.assertEqual(tokens.tolist(), [0, 31414, 232, 328, 2])
            self.assertEqual(bart.decode(tokens), "Hello world!")

            # Extract the last layer's features
            last_layer_features = bart.extract_features(tokens)
            self.assertEqual(last_layer_features.size(), torch.Size([1, 5, 1024]))

            # Extract all layer's features from decoder (layer 0 is the embedding layer)
            all_layers = bart.extract_features(tokens, return_all_hiddens=True)
            self.assertEqual(len(all_layers), 13)
            self.assertTrue(torch.all(all_layers[-1] == last_layer_features))

            # Register a new (randomly initialized) classification head
            bart.register_classification_head("new_task", num_classes=3)
            logprobs = bart.predict("new_task", tokens)  # noqa

    @torch.no_grad()
    def test_bart_large_mnli(self):
        with contextlib.redirect_stdout(StringIO()):
            # Download BART already finetuned for MNLI
            bart = fb_hub.load("bart.large.mnli")
            bart.eval()  # disable dropout for evaluation

            # Encode a pair of sentences and make a prediction
            tokens = bart.encode(
                "BART is a seq2seq model.", "BART is not sequence to sequence."
            )
            prediction = bart.predict("mnli", tokens).argmax().item()
            self.assertEqual(prediction, 0)  # contradiction

            # Encode another pair of sentences
            tokens = bart.encode(
                "BART is denoising autoencoder.", "BART is version of autoencoder."
            )
            prediction = bart.predict("mnli", tokens).argmax().item()
            self.assertEqual(prediction, 2)  # entailment

            # Test batched prediction
            from fairseq.data.data_utils import collate_tokens

            batch_of_pairs = [
                ["BART is a seq2seq model.", "BART is not sequence to sequence."],
                ["BART is denoising autoencoder.", "BART is version of autoencoder."],
            ]
            batch = collate_tokens(
                [bart.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
            )
            logprobs = bart.predict("mnli", batch)
            self.assertEqual(logprobs.argmax(dim=1).tolist(), [0, 2])

    @torch.no_grad()
    def test_bart_large_cnn(self):
        with contextlib.redirect_stdout(StringIO()):
            # Download BART already finetuned for MNLI
            bart = fb_hub.load("bart.large.cnn")
            bart.eval()  # disable dropout for evaluation

            hypothesis = bart.sample(
                [
                    """This is the first time anyone has been \
recorded to run a full marathon of 42.195 kilometers \
(approximately 26 miles) under this pursued landmark time. \
It was not, however, an officially sanctioned world record, \
as it was not an "open race" of the IAAF. His time was \
1 hour 59 minutes 40.2 seconds. Kipchoge ran in Vienna, Austria. \
It was an event specifically designed to help Kipchoge \
break the two hour barrier. Kenyan runner Eliud Kipchoge \
has run a marathon in less than two hours."""
                ]
            )

            # Encode a pair of sentences and make a prediction
            self.assertEqual(
                hypothesis[0],
                """Eliud Kipchoge has run a marathon in less than two hours. \
Kenyan ran in Vienna, Austria. It was not an officially sanctioned world record.""",
            )


if __name__ == "__main__":
    unittest.main()
