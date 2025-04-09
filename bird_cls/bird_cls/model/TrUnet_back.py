import torch
import torchvision
from torch import nn
from einops import rearrange
import torch.nn.functional as F

EPS = 1e-8


class StandardConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv2d, self).__init__()
        self.StandardConv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=stride // 2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.StandardConv2d(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.DepthwiseSeparableConv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=out_channels,
            #           out_channels=out_channels,
            #           kernel_size=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.DepthwiseSeparableConv2d(x)


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, training=False):
    frames = x.split(1, -2)
    m_frames = []
    last_state = None
    for frame in frames:
        if last_state is None:
            last_state = s * frame
            m_frames.append(last_state)
            continue
        if training:
            m_frame = ((1 - s) * last_state).add(s * frame)
        else:
            m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    if training:
        pcen_ = (x / (M + eps).pow(alpha) + delta).pow(r) - delta ** r
    else:
        pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class PCENTransform(nn.Module):
    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, trainable=True):
        super().__init__()
        if trainable:
            self.log_s = nn.Parameter(torch.log(torch.Tensor([s])))
            self.log_alpha = nn.Parameter(torch.log(torch.Tensor([alpha])))
            self.log_delta = nn.Parameter(torch.log(torch.Tensor([delta])))
            self.log_r = nn.Parameter(torch.log(torch.Tensor([r])))
        else:
            self.s = s
            self.alpha = alpha
            self.delta = delta
            self.r = r
        self.eps = eps
        self.trainable = trainable

    def forward(self, x):
        x = x.permute((0, 1, 3, 2)).squeeze(dim=1)
        if self.trainable:
            x = pcen(x, self.eps, torch.exp(self.log_s), torch.exp(self.log_alpha), torch.exp(self.log_delta),
                     torch.exp(self.log_r), self.training and self.trainable)
        else:
            x = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.training and self.trainable)
        x = x.unsqueeze(dim=1).permute((0, 1, 3, 2))
        return x


class GRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, bidirectional):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True, bidirectional=bidirectional)

        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size * (2 if bidirectional == True else 1), out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        output, h = self.GRU(x)
        output = output.transpose(1, 2)
        output = self.conv(output)
        return output


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2,
                                     padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x_last, x):
        x = self.up(x_last) + x
        x = F.leaky_relu(self.norm(x))
        return x


class TrUnet(nn.Module):
    def __init__(self, in_channels=3, cls=2, ):
        super(TrUnet, self).__init__()
        self.in_channels = in_channels
        self.pcen = PCENTransform()
        self.down1 = StandardConv2d(in_channels, 64, 5, 1)
        self.down2 = DepthwiseSeparableConv2d(64, 128, 3, 2)
        self.down3 = DepthwiseSeparableConv2d(128, 256, 3, 1)
        self.down4 = DepthwiseSeparableConv2d(256, 512, 3, 2)
        self.down5 = DepthwiseSeparableConv2d(512, 1024, 3, 2)
        self.down6 = DepthwiseSeparableConv2d(1024, 2048, 3, 2)
        self.fc_features = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(i, 4096), nn.ReLU()) for
             i in [2048, 1024, 512]])
        self.up_conv = nn.ModuleList(
            [UpsampleBlock(in_channels=2048, out_channels=1024),
             UpsampleBlock(in_channels=1024, out_channels=512)],
        )
        # 定义门控单元参数
        self.gate_layer_1 = nn.Linear(4096 * 2, 4096, bias=False)
        self.gate_layer_2 = nn.Linear(4096 * 2, 4096, bias=False)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, cls)
        self.dropout = nn.Dropout(0.5, inplace=True)
        self.norm = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm1d(4096)

    def forward(self, x):
        # pcen_features = self.pcen(x[:, -1, ...].unsqueeze(dim=1))
        # pcen_features = (pcen_features - pcen_features.mean()) / pcen_features.std()
        # features = torch.cat((x[:, :-1, ...], pcen_features), dim=1)
        features = self.norm(x)
        x1 = self.down1(features)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        features = [self.fc_features[0](x6)]
        x = self.up_conv[0](x6, x5)
        features.append(self.fc_features[1](x))
        x = self.up_conv[1](x, x4)
        features.append(self.fc_features[2](x))
        f1 = torch.sigmoid(self.gate_layer_1(torch.cat((features[0], features[1]), dim=1)))
        f2 = torch.sigmoid(self.gate_layer_2(torch.cat((features[0], features[2]), dim=1)))
        features = f1 * features[1] + (1 - f1 * f2) * features[0] + f2 * features[2]
        features = self.norm2(features)
        x9 = self.dropout(F.leaky_relu(self.fc1(features)))
        x10 = self.fc2(x9)
        return x10


if __name__ == '__main__':
    tr_unet = TrUnet(in_channels=1, cls=20)
    x = torch.randn(1, 1, 257, 126)
    y = tr_unet(x)
    pass
