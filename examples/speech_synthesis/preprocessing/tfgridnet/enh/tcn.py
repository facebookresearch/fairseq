# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time–frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
# Licensed under MIT.

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = torch.finfo(torch.get_default_dtype()).eps

class TemporalConvNet(nn.Module):
    def __init__(
        self,
        N,
        B,
        H,
        P,
        X,
        R,
        C,
        Sc=None,
        out_channel=None,
        norm_type="gLN",
        causal=False,
        pre_mask_nonlinear="linear",
        mask_nonlinear="relu",
    ):
        """Basic Module of tasnet.

        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            Sc: Number of channels in skip-connection paths' 1x1-conv blocks
            out_channel: Number of output channels
                if it is None, `N` will be used instead.
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            pre_mask_nonlinear: the non-linear function before masknet
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.skip_connection = Sc is not None
        self.out_channel = N if out_channel is None else out_channel
        if self.skip_connection:
            assert Sc == B, (Sc, B)
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []

        self.receptive_field = 0
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                if r == 0 and x == 0:
                    self.receptive_field += P
                else:
                    self.receptive_field += (P - 1) * dilation
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        Sc,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * self.out_channel, 1, bias=False)
        # Put together (for compatibility with older versions)
        if pre_mask_nonlinear == "linear":
            self.network = nn.Sequential(
                layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1
            )
        else:
            activ = {
                "prelu": nn.PReLU(),
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
            }[pre_mask_nonlinear]
            self.network = nn.Sequential(
                layer_norm, bottleneck_conv1x1, temporal_conv_net, activ, mask_conv1x1
            )

    def forward(self, mixture_w):
        """Keep this API same with TasNet.

        Args:
            mixture_w: [M, N, K], M is batch size

        Returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        bottleneck = self.network[:2]
        tcns = self.network[2]
        masknet = self.network[3:]
        output = bottleneck(mixture_w)
        skip_conn = 0.0
        for block in tcns:
            for layer in block:
                tcn_out = layer(output)
                if self.skip_connection:
                    residual, skip = tcn_out
                    skip_conn = skip_conn + skip
                else:
                    residual = tcn_out
                output = output + residual
        # Use residual output when no skip connection
        if self.skip_connection:
            score = masknet(skip_conn)
        else:
            score = masknet(output)

        # [M, C*self.out_channel, K] -> [M, C, self.out_channel, K]
        score = score.view(M, self.C, self.out_channel, K)
        if self.mask_nonlinear == "softmax":
            est_mask = torch.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = torch.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = torch.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = torch.tanh(score)
        elif self.mask_nonlinear == "linear":
            est_mask = score
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask

class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        self.skip_connection = skip_channels is not None
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            skip_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """Forward.

        Args:
            x: [M, B, K]

        Returns:
            [M, B, K]
        """
        if self.skip_connection:
            res_out, skip_out = self.net(x)
            return res_out, skip_out
        else:
            res_out = self.net(x)
            return res_out

def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = choose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

        # skip connection
        if skip_channels is not None:
            self.skip_conv = nn.Conv1d(in_channels, skip_channels, 1, bias=False)
        else:
            self.skip_conv = None

    def forward(self, x):
        """Forward.

        Args:
            x: [M, H, K]

        Returns:
            res_out: [M, B, K]
            skip_out: [M, Sc, K]
        """
        shared_block = self.net[:-1]
        shared = shared_block(x)
        res_out = self.net[-1](shared)
        if self.skip_conv is None:
            return res_out
        skip_out = self.skip_conv(shared)
        return res_out, skip_out



class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length

        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y