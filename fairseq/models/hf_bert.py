from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from . import (
    FairseqIncrementalDecoder, FairseqLanguageModel, register_model,
    register_model_architecture
)

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 sparse_transformer=False,
                 is_bidirectional=True,
                 stride=32,
                 expressivity=8,
                 ):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.sparse_transformer = sparse_transformer
            self.is_bidirectional = is_bidirectional
            self.stride = stride
            self.expressivity = expressivity
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.config = config
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        assert not torch.isnan(words_embeddings).any()
        position_embeddings = self.position_embeddings(position_ids)
        assert not torch.isnan(position_embeddings).any()
        token_type_embeddings = self.token_type_embeddings(token_type_ids.long())
        assert not torch.isnan(token_type_embeddings).any()
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sparse_transformer = config.sparse_transformer
        self.is_bidirectional = config.is_bidirectional
        self.stride = config.stride
        self.expressivity = config.expressivity
        self.max_position_embeddings =  config.max_position_embeddings
        if self.sparse_transformer:
            self.prepare_sparse_mask()

    def prepare_sparse_mask(self):
        print("prepare sparse mask")
        sparse_mask = torch.empty((self.max_position_embeddings, self.max_position_embeddings)).float().fill_(-10000.0)

        # If bidirectional, subset 2 is the same for every index
        subset_summaries = set()
        if self.is_bidirectional:
            subset_summaries = self.compute_subset_summaries(self.max_position_embeddings)

        for i in range(self.max_position_embeddings):
            fixed_attention_subset = self.compute_fixed_attention_subset(i, self.max_position_embeddings)
            fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
            included_word_indices = torch.LongTensor(list(fixed_attention_subset))
            sparse_mask[i].index_fill_(0, included_word_indices, 0)
        self.register_buffer("sparse_mask", sparse_mask)
        print("initialize sparse mask of shape", self.sparse_mask.size())
        print("first row contains %d nnz" % (self.sparse_mask[0] == 0).sum())
        print("last row contains %d nnz" % (self.sparse_mask[-1] == 0).sum())

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        # batch_size x num_attention_heads x length x d_head

    # Used for Ai(2) calculations - beginning of [l-c, l] range
    def compute_checkpoint(self, word_index):
        if word_index % self.stride == 0 and word_index != 0:
            checkpoint_index = word_index - self.expressivity
        else:
            checkpoint_index = (
                math.floor(word_index / self.stride) * self.stride
                + self.stride - self.expressivity
            )
        return checkpoint_index

    # Computes Ai(2)
    def compute_subset_summaries(self, absolute_max):
        checkpoint_index = self.compute_checkpoint(0)
        subset_two = set()
        while checkpoint_index <= absolute_max-1:
            summary = set(range(checkpoint_index, min(
                checkpoint_index+self.expressivity+1, absolute_max)
            ))
            subset_two = subset_two.union(summary)
            checkpoint_index = self.compute_checkpoint(checkpoint_index+self.stride)
        return subset_two

    # Sparse Transformer Fixed Attention Pattern: https://arxiv.org/pdf/1904.10509.pdf
    def compute_fixed_attention_subset(self, word_index, tgt_len):
        # +1s account for range function; [min, max) -> [min, max]
        if not self.is_bidirectional:
            absolute_max = word_index + 1
        else:
            absolute_max = tgt_len

        # Subset 1 - whole window
        rounded_index = math.floor((word_index + self.stride) / self.stride) * self.stride
        if word_index % self.stride == 0 and word_index != 0:
            subset_one = set(range(word_index-self.stride, min(absolute_max, word_index+1)))
        else:
            subset_one = set(range(max(0, rounded_index - self.stride), min(
                absolute_max, rounded_index+1))
            )

        # Subset 2 - summary per window
        # If bidirectional, subset 2 is the same for every index
        subset_two = set()
        if not self.is_bidirectional:
            subset_two = self.compute_subset_summaries(absolute_max)

        return subset_one.union(subset_two)

    # Compute sparse mask - if bidirectional, can pre-compute and store
    def buffered_sparse_mask(self, tensor, tgt_len, src_len):
        return self.sparse_mask[:tgt_len, :src_len].type_as(tensor)
        assert(self.max_position_embeddings > self.stride)
        #  assert(tgt_len > self.stride)
        #  sparse_mask = torch.empty((tgt_len, src_len)).float().fill_(float('-inf'))

        if self.sparse_mask is None:
            sparse_mask = torch.empty((self.max_position_embeddings, self.max_position_embeddings)).float().fill_(-10000.0)

            # If bidirectional, subset 2 is the same for every index
            subset_summaries = set()
            if self.is_bidirectional:
                subset_summaries = self.compute_subset_summaries(self.max_position_embeddings)

            for i in range(self.max_position_embeddings):
                fixed_attention_subset = self.compute_fixed_attention_subset(i, self.max_position_embeddings)
                fixed_attention_subset = fixed_attention_subset.union(subset_summaries)
                included_word_indices = torch.LongTensor(list(fixed_attention_subset))
                sparse_mask[i].index_fill_(0, included_word_indices, 0)
            self.sparse_mask = sparse_mask.type_as(tensor)
            print("initialize sparse mask of shape", self.sparse_mask.size())
            print("first row contains %d nnz" % (self.sparse_mask[0] == 0).sum())
            print("last row contains %d nnz" % (self.sparse_mask[-1] == 0).sum())

        return self.sparse_mask[:tgt_len, :src_len]

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        if not self.sparse_transformer:
            return attn_weights
        sparse_mask = self.buffered_sparse_mask(attn_weights, tgt_len, src_len)
        #  sparse_mask = sparse_mask.unsqueeze(0).expand(bsz * self.num_attention_heads, tgt_len, src_len)
        attn_weights += sparse_mask
        return attn_weights


    def forward(self, hidden_states, attention_mask):
        batch_size, length = hidden_states.size()[:2]

        mixed_query_layer = self.query(hidden_states)
        assert not torch.isnan(mixed_query_layer).any()
        mixed_key_layer = self.key(hidden_states)
        assert not torch.isnan(mixed_key_layer).any()
        mixed_value_layer = self.value(hidden_states)
        assert not torch.isnan(mixed_value_layer).any()

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        assert not torch.isnan(attention_scores).any()
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        assert not torch.isnan(attention_scores).any()

        attention_scores = self.apply_sparse_mask(attention_scores, length, length, batch_size)

        attention_scores = torch.clamp(attention_scores, -10000., 10000.)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)
        #assert not torch.isnan(attention_probs).any()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        #assert not torch.isnan(hidden_states).any()
        hidden_states = self.dropout(hidden_states)
        #assert not torch.isnan(hidden_states).any()
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #assert not torch.isnan(hidden_states).any()
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        #assert not torch.isnan(self_output).any()
        attention_output = self.output(self_output, input_tensor)
        #assert not torch.isnan(attention_output).any()
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        #assert not torch.isnan(hidden_states).any()
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #assert not torch.isnan(hidden_states).any()
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        #assert not torch.isnan(attention_output).any()
        intermediate_output = self.intermediate(attention_output)
        #assert not torch.isnan(intermediate_output).any()
        layer_output = self.output(intermediate_output, attention_output)
        #assert not torch.isnan(layer_output).any()
        return layer_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, remove_head=False, remove_pooled=False):
        super(BertModel, self).__init__(config, remove_head)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if not remove_head:
            self.pooler = BertPooler(config)
        else:
            if not remove_pooled:
                self.pooler = BertPooler(config)
            else:
                self.pooler = None
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        assert not (extended_attention_mask.sum(-1) == 0).any()
        #extended_mask = extended_attention_mask.new(extended_attention_mask.size(0), extended_attention_mask.size(1)).fill_(-math.inf).float()
        #extended_mask[extended_attention_mask.byte()] = 0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        assert not torch.isnan(embedding_output).any()
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        #assert not torch.isnan(layer).any()
        sequence_output = encoded_layers[-1]
        assert not torch.isnan(sequence_output).any()
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output)
            assert not torch.isnan(pooled_output).any()
        else:
            pooled_output = None
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

@register_model('bert_hf')
class BertHF(FairseqLanguageModel):
    def __init__(self, decoder, task):
        super().__init__(decoder)
        self.task = task

    def forward(self, src_tokens, segment_labels, **unused):
        padding_mask = src_tokens.ne(self.task.dictionary.pad())
        return self.decoder(input_ids=src_tokens, token_type_ids=segment_labels, attention_mask=padding_mask)

    def max_positions(self):
        return self.decoder.config.max_position_embeddings

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--hidden-size', type=int,
                help='decoder embedding dimension')
        parser.add_argument('--num-hidden-layers', type=int,
                help='num decoder layers')
        parser.add_argument('--num-attention-heads', type=int,
                help='num decoder attention heads')
        parser.add_argument('--intermediate-size', type=int,
                help='decoder embedding dimension for FFN')
        parser.add_argument('--hidden_act', default='gelu', type=str,
                help='activation function type')
        parser.add_argument('--attention-probs-dropout-prob', type=float,
                help='dropout probability for attention weights')
        parser.add_argument('--hidden-dropout-prob', type=float,
                help='dropout probability')
        parser.add_argument('--max-position-embeddings', default=512, type=int,
                help='sequence length')
        parser.add_argument('--initializer-range', type=float,
                help='initializer std')
    @classmethod
    def build_model(cls, args, task):
        args.remove_head = getattr(args, 'remove_head', False)
        args.remove_pooled = getattr(args, 'remove_pooled', False)
        decoder = BertForPreTraining(args.config, args.remove_head, args.remove_pooled)
        return BertHF(decoder, task)


@register_model('simple_bert_hf')
class SimpleBertHF(FairseqLanguageModel):
    def __init__(self, decoder, task):
        super().__init__(decoder)
        self.task = task

    def forward(self, src_tokens, segment_labels, **unused):
        padding_mask = src_tokens.ne(self.task.dictionary.pad())
        return self.decoder(input_ids=src_tokens, token_type_ids=segment_labels, attention_mask=padding_mask)

    def max_positions(self):
        return self.decoder.config.max_position_embeddings

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--hidden-size', type=int,
                help='decoder embedding dimension')
        parser.add_argument('--num-hidden-layers', type=int,
                help='num decoder layers')
        parser.add_argument('--num-attention-heads', type=int,
                help='num decoder attention heads')
        parser.add_argument('--intermediate-size', type=int,
                help='decoder embedding dimension for FFN')
        parser.add_argument('--hidden_act', default='gelu', type=str,
                help='activation function type')
        parser.add_argument('--attention-probs-dropout-prob', type=float,
                help='dropout probability for attention weights')
        parser.add_argument('--hidden-dropout-prob', type=float,
                help='dropout probability')
        parser.add_argument('--max-position-embeddings', default=512, type=int,
                help='sequence length')
        parser.add_argument('--initializer-range', type=float,
                help='initializer std')
        parser.add_argument('--sparse-transformer', default=False, action='store_true',
                help='whether to use sparse transformer')
        parser.add_argument('--sparse-transformer-stride', default=32, type=int,
                help='sparse transformer stride')
        parser.add_argument('--sparse-transformer-expressivity', default=8, type=int,
                help='sparse transformer expressivity')
    @classmethod
    def build_model(cls, args, task):
        args.remove_only_mlm_head = getattr(args, 'remove_only_mlm_head', False)
        decoder = SimpleBertForPreTraining(args.config, args.remove_only_mlm_head)
        return SimpleBertHF(decoder, task)

@register_model_architecture('simple_bert_hf', 'simple_sparse_bert_hf')
def simple_base_bert_architecture(args):
    args.config = BertConfig()
    args.config.hidden_size = getattr(args, 'hidden_size', args.config.hidden_size)
    args.config.num_hidden_layers = getattr(args, 'num_hidden_layers', args.config.num_hidden_layers)
    args.config.num_attention_heads = getattr(args, 'num_attention_heads', args.config.num_attention_heads)
    args.config.intermediate_size = getattr(args, 'intermediate_size', args.config.intermediate_size)
    args.config.hidden_act = getattr(args, 'hidden_act', args.config.hidden_act)
    args.config.hidden_dropout_prob = getattr(args, 'hidden_dropout_prob', args.config.hidden_dropout_prob)
    args.config.attention_probs_dropout_prob = getattr(args, 'attention_probs_dropout_prob', args.config.attention_probs_dropout_prob)
    args.config.max_position_embeddings = getattr(args, 'max_position_embeddings', args.config.max_position_embeddings)
    args.config.initializer_range = getattr(args, 'initializer_range', args.config.initializer_range)
    args.config.sparse_transformer = getattr(args, 'sparse_transformer', args.config.sparse_transformer)
    args.config.is_bidirectional = True
    args.config.stride = getattr(args, 'sparse_transformer_stride', args.config.stride)
    args.config.expressivity = getattr(args, 'sparse_transformer_expressivity', args.config.expressivity)

@register_model_architecture('simple_bert_hf', 'simple_bert_hf')
def simple_base_bert_architecture(args):
    args.config = BertConfig()
    args.config.hidden_size = getattr(args, 'hidden_size', args.config.hidden_size)
    args.config.num_hidden_layers = getattr(args, 'num_hidden_layers', args.config.num_hidden_layers)
    args.config.num_attention_heads = getattr(args, 'num_attention_heads', args.config.num_attention_heads)
    args.config.intermediate_size = getattr(args, 'intermediate_size', args.config.intermediate_size)
    args.config.hidden_act = getattr(args, 'hidden_act', args.config.hidden_act)
    args.config.hidden_dropout_prob = getattr(args, 'hidden_dropout_prob', args.config.hidden_dropout_prob)
    args.config.attention_probs_dropout_prob = getattr(args, 'attention_probs_dropout_prob', args.config.attention_probs_dropout_prob)
    args.config.max_position_embeddings = getattr(args, 'max_position_embeddings', args.config.max_position_embeddings)
    args.config.initializer_range = getattr(args, 'initializer_range', args.config.initializer_range)

@register_model_architecture('simple_bert_hf', 'simple_bert_hf_large')
def simple_large_bert_architecture(args):
    args.config = BertConfig()
    args.config.hidden_size = getattr(args, 'hidden_size', 1024)
    args.config.num_hidden_layers = getattr(args, 'num_hidden_layers', 24)
    args.config.num_attention_heads = getattr(args, 'num_attention_heads', 16)
    args.config.intermediate_size = getattr(args, 'intermediate_size', 4096)
    args.config.hidden_act = getattr(args, 'hidden_act', 'gelu')
    args.config.hidden_dropout_prob = getattr(args, 'hidden_dropout_prob', 0.1)
    args.config.attention_probs_dropout_prob = getattr(args, 'attention_probs_dropout_prob', 0.1)
    args.config.max_position_embeddings = getattr(args, 'max_position_embeddings', 512)
    args.config.initializer_range = getattr(args, 'initializer_range', 0.02)


@register_model_architecture('bert_hf', 'bert_hf')
def base_bert_architecture(args):
    args.config = BertConfig()
    args.config.hidden_size = getattr(args, 'hidden_size', args.config.hidden_size)
    args.config.num_hidden_layers = getattr(args, 'num_hidden_layers', args.config.num_hidden_layers)
    args.config.num_attention_heads = getattr(args, 'num_attention_heads', args.config.num_attention_heads)
    args.config.intermediate_size = getattr(args, 'intermediate_size', args.config.intermediate_size)
    args.config.hidden_act = getattr(args, 'hidden_act', args.config.hidden_act)
    args.config.hidden_dropout_prob = getattr(args, 'hidden_dropout_prob', args.config.hidden_dropout_prob)
    args.config.attention_probs_dropout_prob = getattr(args, 'attention_probs_dropout_prob', args.config.attention_probs_dropout_prob)
    args.config.max_position_embeddings = getattr(args, 'max_position_embeddings', args.config.max_position_embeddings)
    args.config.initializer_range = getattr(args, 'initializer_range', args.config.initializer_range)

@register_model_architecture('bert_hf', 'bert_hf_large')
def large_bert_architecture(args):
    args.config = BertConfig()
    args.config.hidden_size = getattr(args, 'hidden_size', 1024)
    args.config.num_hidden_layers = getattr(args, 'num_hidden_layers', 24)
    args.config.num_attention_heads = getattr(args, 'num_attention_heads', 16)
    args.config.intermediate_size = getattr(args, 'intermediate_size', 4096)
    args.config.hidden_act = getattr(args, 'hidden_act', 'gelu')
    args.config.hidden_dropout_prob = getattr(args, 'hidden_dropout_prob', 0.1)
    args.config.attention_probs_dropout_prob = getattr(args, 'attention_probs_dropout_prob', 0.1)
    args.config.max_position_embeddings = getattr(args, 'max_position_embeddings', 512)
    args.config.initializer_range = getattr(args, 'initializer_range', 0.02)

class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, remove_head = False, remove_pooled=False):
        super(BertForPreTraining, self).__init__(config, remove_head)
        self.config = config
        self.bert = BertModel(config, remove_head=remove_head, remove_pooled=remove_pooled)
        self.remove_head = remove_head
        if not remove_head:
            self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        max_position_embeddings = self.config.max_position_embeddings
        input_ids = input_ids[:, :max_position_embeddings]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :max_position_embeddings]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :max_position_embeddings]
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        if self.remove_head:
            return sequence_output, pooled_output
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss

        return prediction_scores, seq_relationship_score

class SimpleBertForPreTraining(PreTrainedBertModel):
    def __init__(self, config, remove_only_mlm_head=False):
        super(SimpleBertForPreTraining, self).__init__(config)
        self.config = config
        self.bert = BertModel(config, remove_head=True, remove_pooled=True)
        self.remove_only_mlm_head = remove_only_mlm_head

        if not remove_only_mlm_head:
            self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        max_position_embeddings = self.config.max_position_embeddings
        input_ids = input_ids[:, :max_position_embeddings]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :max_position_embeddings]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :max_position_embeddings]
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        if self.remove_only_mlm_head:
            return sequence_output

        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores
