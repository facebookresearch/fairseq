import torch
import torch.nn as nn
import math
import torch.utils.serialization
import torch.nn.functional as F


class FConvModel(nn.Module):
    def __init__(self, datasets, encoder, decoder):
        super(FConvModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer('mask', torch.ones(len(datasets.dst_dict)))
        self.mask[datasets.dst_dict.index('<pad>')] = 0

    def forward(self, src_tokens, src_positions, input_tokens, input_positions, target, ntokens):
        encoder_out = self.encoder(src_tokens, src_positions)
        decoder_out = self.decoder(input_tokens, input_positions, encoder_out)
        decoder_out = decoder_out.view(-1, decoder_out.size(-1))
        target = target.view(-1)
        loss = F.cross_entropy(decoder_out, target, self.mask, size_average=False)
        return loss / ntokens


class Encoder(nn.Module):
    """Convolutional encoder"""
    def __init__(self, num_embeddings, embed_dim=512, max_position=1024,
                 convolutions=((512, 3),) * 20, dropout=0.1, padding_idx=1,
                 num_attention_layers=1):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.num_attention_layers = num_attention_layers
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_position, embed_dim, padding_idx)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = (kernel_size - 1) // 2
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size, padding=pad,
                       dropout=dropout))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, tokens, positions):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else self.proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x.transpose(1, 2)).transpose(2, 1)
            x = F.glu(x)
            x = (x + residual) * math.sqrt(0.5)

        # project back to size of embedding
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        x = grad_multiply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return x, y


class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

    def forward(self, x, target_embedding, encoder_out):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = torch.bmm(x, encoder_out[0].transpose(1, 2))

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)

        x = torch.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x


class Decoder(nn.Module):
    """Convolutional decoder"""
    def __init__(self, num_embeddings=24638, embed_dim=512, out_embed_dim=256,
                 max_position=1024, convolutions=((512, 3),) * 20,
                 dropout=0.1, padding_idx=1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        in_channels = convolutions[0][0]

        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = Embedding(max_position, embed_dim, padding_idx)
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        for (out_channels, kernel_size) in convolutions:
            pad = kernel_size - 1
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                Conv1d(in_channels, out_channels * 2, kernel_size, padding=pad,
                       dropout=dropout))
            self.attention.append(AttentionLayer(out_channels, embed_dim))
            in_channels = out_channels
        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings)

    def forward(self, tokens, positions, encoder_out):
        # embed tokens and positions
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # temporal convolutions
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else self.proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x.transpose(1, 2)).transpose(2, 1)
            x = x[:, :-conv.padding[0], :]  # remove future timestamps
            x = F.glu(x)

            # attention
            x = attention(x, target_embedding, encoder_out)

            # residual
            x = (x + residual) * math.sqrt(0.5)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x

    def context_size(self):
        """Maximum number of input elements each output element depends on"""
        context = 1
        for conv in self.convolutions:
            context += conv.kernel_size[0] - 1
        return context


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    # m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    return nn.utils.weight_norm(m)


def grad_multiply(x, scale):
    return GradMultiply.apply(x, scale)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def fconv_iwslt_de_en(dataset, dropout):
    padding_idx = dataset.dst_dict.index('<pad>')

    encoder = Encoder(
        len(dataset.src_dict),
        embed_dim=256,
        convolutions=((256, 3),) * 4,
        dropout=dropout,
        padding_idx=padding_idx,
        num_attention_layers=3)
    decoder = Decoder(
        len(dataset.dst_dict),
        embed_dim=256,
        convolutions=((256, 3),) * 3,
        dropout=dropout,
        padding_idx=padding_idx)
    return FConvModel(dataset, encoder, decoder)


def fconv_wmt_en_ro(dataset, dropout):
    padding_idx = dataset.dst_dict.index('<pad>')

    encoder = Encoder(
        len(dataset.src_dict),
        embed_dim=512,
        convolutions=((512, 3),) * 20,
        dropout=dropout,
        padding_idx=padding_idx,
        num_attention_layers=3)
    decoder = Decoder(
        len(dataset.dst_dict),
        embed_dim=512,
        convolutions=((512, 3),) * 20,
        dropout=dropout,
        padding_idx=padding_idx)
    return FConvModel(dataset, encoder, decoder)
