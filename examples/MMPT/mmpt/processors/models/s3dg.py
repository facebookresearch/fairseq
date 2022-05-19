# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Contains a PyTorch definition for Gated Separable 3D network (S3D-G)
with a text module for computing joint text-video embedding from raw text
and video input. The following code will enable you to load the HowTo100M
pretrained S3D Text-Video model from:
  A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic and A. Zisserman,
  End-to-End Learning of Visual Representations from Uncurated Instructional Videos.
  https://arxiv.org/abs/1912.06430.

S3D-G was proposed by:
  S. Xie, C. Sun, J. Huang, Z. Tu and K. Murphy,
  Rethinking Spatiotemporal Feature Learning For Video Understanding.
  https://arxiv.org/abs/1712.04851.
  Tensorflow code: https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py

The S3D architecture was slightly modified with a space to depth trick for TPU
optimization.
"""

import torch as th
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import re


class InceptionBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        num_outputs_0_0a,
        num_outputs_1_0a,
        num_outputs_1_0b,
        num_outputs_2_0a,
        num_outputs_2_0b,
        num_outputs_3_0b,
        gating=True,
    ):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(
            num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3], padding=1, separable=True
        )
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(
            num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3], padding=1, separable=True
        )
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = (
            num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b
        )
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
        """Inception block
      """
        b0 = self.conv_b0(input)
        b1 = self.conv_b1_a(input)
        b1 = self.conv_b1_b(b1)
        b2 = self.conv_b2_a(input)
        b2 = self.conv_b2_b(b2)
        b3 = self.maxpool_b3(input)
        b3 = self.conv_b3_b(b3)
        if self.gating:
            b0 = self.gating_b0(b0)
            b1 = self.gating_b1(b1)
            b2 = self.gating_b2(b2)
            b3 = self.gating_b3(b3)
        return th.cat((b0, b1, b2, b3), dim=1)


class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G.
      """
        spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = th.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class STConv3D(nn.Module):
    def __init__(
        self, input_dim, output_dim, kernel_size, stride=1, padding=0, separable=False
    ):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
                spatial_stride = [1, stride[1], stride[2]]
                temporal_stride = [stride[0], 1, 1]
            else:
                spatial_stride = [1, stride, stride]
                temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
                spatial_padding = [0, padding[1], padding[2]]
                temporal_padding = [padding[0], 0, 0]
            else:
                spatial_padding = [0, padding, padding]
                temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=spatial_kernel_size,
                stride=spatial_stride,
                padding=spatial_padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(
                output_dim,
                output_dim,
                kernel_size=temporal_kernel_size,
                stride=temporal_stride,
                padding=temporal_padding,
                bias=False,
            )
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn1 = nn.BatchNorm3d(output_dim)

    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


class MaxPool3dTFPadding(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = self._get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def _get_padding_shape(self, filter_shape, stride):
        def _pad_top_bottom(filter_dim, stride_val):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            return pad_top, pad_bottom

        padding_shape = []
        for filter_dim, stride_val in zip(filter_shape, stride):
            pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        return tuple(padding_shape)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Sentence_Embedding(nn.Module):
    def __init__(
        self,
        embd_dim,
        num_embeddings=66250,
        word_embedding_dim=300,
        token_to_word_path="dict.npy",
        max_words=16,
        output_dim=2048,
    ):
        super(Sentence_Embedding, self).__init__()
        self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, embd_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [
            self.word_to_token[word] for word in words if word in self.word_to_token
        ]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent.lower())) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x):
        x = self._words_to_ids(x)
        x = self.word_embd(x)
        x = F.relu(self.fc1(x))
        x = th.max(x, dim=1)[0]
        x = self.fc2(x)
        return {'text_embedding': x}


class S3D(nn.Module):
    def __init__(self, dict_path, num_classes=512, gating=True, space_to_depth=True):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        if space_to_depth:
            self.conv1 = STConv3D(
                24, 64, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False
            )
        else:
            self.conv1 = STConv3D(
                3, 64, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False
            )
        self.conv_2b = STConv3D(64, 64, [1, 1, 1], separable=False)
        self.conv_2c = STConv3D(64, 192, [3, 3, 3], padding=1, separable=True)
        self.gating = SelfGating(192)
        self.maxpool_2a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.maxpool_3a = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME"
        )
        self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = InceptionBlock(
            self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64
        )
        self.maxpool_4a = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_4b = InceptionBlock(
            self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64
        )
        self.mixed_4c = InceptionBlock(
            self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64
        )
        self.mixed_4d = InceptionBlock(
            self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64
        )
        self.mixed_4e = InceptionBlock(
            self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64
        )
        self.mixed_4f = InceptionBlock(
            self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME"
        )
        self.mixed_5b = InceptionBlock(
            self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128
        )
        self.mixed_5c = InceptionBlock(
            self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128
        )
        self.fc = nn.Linear(self.mixed_5c.output_dim, num_classes)
        self.text_module = Sentence_Embedding(num_classes,
            token_to_word_path=dict_path)

    def _space_to_depth(self, input):
        """3D space to depth trick for TPU optimization.
      """
        B, C, T, H, W = input.shape
        input = input.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
        input = input.permute(0, 3, 5, 7, 1, 2, 4, 6)
        input = input.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
        return input

    def forward(self, inputs):
        """Defines the S3DG base architecture."""
        if self.space_to_depth:
            inputs = self._space_to_depth(inputs)
        net = self.conv1(inputs)
        if self.space_to_depth:
            # we need to replicate 'SAME' tensorflow padding
            net = net[:, :, 1:, 1:, 1:]
        net = self.maxpool_2a(net)
        net = self.conv_2b(net)
        net = self.conv_2c(net)
        if self.gating:
            net = self.gating(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net = self.mixed_5c(net)
        net = th.mean(net, dim=[2, 3, 4])
        return {'video_embedding': self.fc(net), 'mixed_5c': net}
