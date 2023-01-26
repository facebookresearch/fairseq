import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


class CpcFeatureReader:
    """
    Wrapper class to run inference on CPC model.
    Helps extract features for a given audio file.
    """

    def __init__(
        self,
        checkpoint_path,
        layer,
        use_encoder_layer=False,
        norm_features=False,
        sample_rate=16000,
        max_chunk=64000,
        use_cuda=True,
    ):
        self.model = load_cpc_model(checkpoint_path, layer).eval()
        self.sample_rate = sample_rate
        self.max_chunk = max_chunk
        self.norm_features = norm_features
        self.use_encoder_layer = use_encoder_layer
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def read_audio(self, path, ref_len=None, channel_id=None):
        wav, sr = sf.read(path)
        if channel_id is not None:
            assert wav.ndim == 2, \
                f"Expected stereo input when channel_id is given ({path})"
            assert channel_id in [1, 2], \
                "channel_id is expected to be in [1, 2]"
            wav = wav[:, channel_id-1]
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None, channel_id=None):
        x = self.read_audio(file_path, ref_len, channel_id)
        # Inspired from CPC_audio feature_loader.py
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            if self.use_cuda:
                x = x.cuda()
            x = x.view(1, 1, -1)
            size = x.size(2)
            feat = []
            start = 0
            while start < size:
                if start + self.max_chunk > size:
                    break
                x_chunk = x[..., start : start + self.max_chunk]
                feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    get_encoded=self.use_encoder_layer,
                    norm_output=self.norm_features,
                )
                feat.append(feat_chunk)
                start += self.max_chunk

            if start < size:
                x_chunk = x[:, -self.max_chunk :]
                feat_chunk = self.model.extract_features(
                    source=x_chunk,
                    get_encoded=self.use_encoder_layer,
                    norm_output=self.norm_features,
                )
                df = x_chunk.size(2) // feat_chunk.size(1)
                delta = (size - start) // df
                feat.append(feat_chunk[:, -delta:])
        return torch.cat(feat, 1).squeeze(0)


def load_cpc_model(checkpoint_path, layer=None):
    state_dict = torch.load(checkpoint_path)
    weights = state_dict["weights"]
    config = state_dict["config"]
    if layer is not None:
        config["nLevelsGRU"] = layer

    encoder = CPCEncoder(config["hiddenEncoder"])
    ar_net = CPCAR(
        config["hiddenEncoder"], config["hiddenGar"], False, config["nLevelsGRU"]
    )

    model = CPCModel(encoder, ar_net)
    model.load_state_dict(weights, strict=False)
    model.config = config

    return model


class ChannelNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-05, affine=True):
        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        cum_mean = x.mean(dim=1, keepdim=True)
        cum_var = x.var(dim=1, keepdim=True)
        x = (x - cum_mean) * torch.rsqrt(cum_var + self.epsilon)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super(CPCEncoder, self).__init__()
        self.conv0 = nn.Conv1d(1, hidden_dim, 10, stride=5, padding=3)
        self.batchNorm0 = ChannelNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 8, stride=4, padding=2)
        self.batchNorm1 = ChannelNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.batchNorm2 = ChannelNorm(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.batchNorm3 = ChannelNorm(hidden_dim)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.batchNorm4 = ChannelNorm(hidden_dim)
        self.DOWNSAMPLING = 160

    def get_output_dim(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):
    def __init__(self, dim_encoded, dim_output, keep_hidden, num_layers):
        super(CPCAR, self).__init__()
        self.baseNet = nn.LSTM(
            dim_encoded, dim_output, num_layers=num_layers, batch_first=True
        )
        self.hidden = None
        self.keep_hidden = keep_hidden

    def get_output_dim(self):
        return self.baseNet.hidden_size

    def forward(self, x):
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keep_hidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()
        return x


class CPCModel(nn.Module):
    def __init__(self, encoder, ar_net):
        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = ar_net
        self.config = None

    def forward(self, x, label):
        encoded = self.gEncoder(x).permute(0, 2, 1)
        cpc_feature = self.gAR(encoded)
        return cpc_feature, encoded, label

    def extract_features(self, source, get_encoded=False, norm_output=False):
        cpc_feature, encoded, _ = self.forward(source, None)
        if get_encoded:
            cpc_feature = encoded
        if norm_output:
            mean = cpc_feature.mean(dim=1, keepdim=True)
            var = cpc_feature.var(dim=1, keepdim=True)
            cpc_feature = (cpc_feature - mean) / torch.sqrt(var + 1e-08)
        return cpc_feature
