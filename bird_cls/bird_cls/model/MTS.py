import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


def positional_encoding(seq_len, d_model):
    """
    seq_len: 序列长度（例如，文本中的词数）
    d_model: 模型的维度（通常是嵌入层的维度）
    """
    # 初始化位置编码矩阵
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # (1, d_model)

    # 计算位置编码
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))  # (seq_len, d_model)
    # print(angles.shape)
    # 对应的正弦和余弦编码
    encoding = np.zeros((seq_len, d_model))
    encoding[:, 0::2] = np.sin(angles[:, 0::2])  # 对应偶数维度使用正弦
    encoding[:, 1::2] = np.cos(angles[:, 1::2])  # 对应奇数维度使用余弦

    return torch.from_numpy(encoding)


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(GlobalLayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 计算整个层的均值和方差
        mean = torch.mean(x)
        var = torch.var(x)

        # 归一化处理
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # 应用缩放和偏移参数
        return self.weight * x_normalized + self.bias


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = SEAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      groups=out_channels,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.pointwise_conv(x)
        out = self.depthwise_conv(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False,
                      ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class Identity(nn.Module):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction_ratio=8):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        self.pointwise_conv = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels,
                                                       stride=stride)
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.cbam = CBAM(in_channels=out_channels, reduction_ratio=reduction_ratio)

        if stride == 2 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down_sample = Identity()
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        x1 = self.pointwise_conv(x)
        x2 = self.conv(x)
        x3 = x1 + x2
        x3 = self.cbam(x3)
        identity = self.down_sample(residual)
        x = self.relu(x3 + identity)
        return x


# 多层感知机
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_dim_list):
        super(MultiScaleFeatureFusion, self).__init__()
        self.mlp_layers = nn.ModuleList()

        # 创建多个MLP层，每个MLP层对应一个尺度的特征
        for i in range(len(input_dim_list) - 1):
            input_dim = input_dim_list[i]
            next_input_dim = input_dim_list[i + 1]
            mlp_layer = nn.Linear(input_dim, next_input_dim, bias=False, )
            self.mlp_layers.append(mlp_layer)
        self.mlp_layers.append(
            nn.Linear(input_dim_list[0], input_dim_list[-1], bias=False)
        )
        self.mlp_layers.append(
            nn.Linear(input_dim_list[1], input_dim_list[-1], bias=False)
        )

    def forward(self, feature_list):
        # 对每个尺度的特征进行MLP处理
        mlp_outputs = None
        for i in range(len(feature_list) - 1):
            feature, mlp_layer = feature_list[i], self.mlp_layers[i]
            if mlp_outputs is None:
                mlp_outputs = feature
            next_feature = feature_list[i + 1]
            out = mlp_layer(mlp_outputs)
            if out.size() == feature_list[i + 1].size():
                mlp_outputs = F.sigmoid(out) * next_feature + next_feature
                mlp_outputs = F.gelu(mlp_outputs)
        # 对作差的进行融合
        mlp_outputs = F.sigmoid(self.mlp_layers[-2](feature_list[0])) * mlp_outputs + mlp_outputs + F.sigmoid(
            self.mlp_layers[-1](feature_list[1]))
        mlp_outputs = F.gelu(mlp_outputs)
        return mlp_outputs


class MultiScaleFeatureFusionNEW(nn.Module):
    def __init__(self, input_dim_list=[3, 2, 0], output_dim=512):
        super(MultiScaleFeatureFusionNEW, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, kernel_size=1, stride=1, padding=0, out_channels=512),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(512, kernel_size=(3, 4), stride=1, padding=0, out_channels=output_dim, groups=512),
        )
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu = nn.GELU()
        self.input_dim_list = input_dim_list

    def forward(self, avg_features):
        avg_pad_features = []
        max_pad_features = []
        for i in range(len(avg_features)):
            b, t = avg_features[i].size()
            # bilinear_feature = F.adaptive_max_pool2d(features[i],output_size=1).flatten(1)
            pad_feature = avg_features[i].reshape(b, 128, 1, -1)
            pad_feature = F.pad(pad_feature, (0, self.input_dim_list[i], 0, 0), 'constant', 0)
            avg_pad_features.append(pad_feature)
        pad_features = torch.concat(avg_pad_features, dim=2)
        avg_x = self.relu(self.bn1(self.conv1(pad_features))).flatten(start_dim=1)
        return avg_x


class Multi_Cls(nn.Module):
    def __init__(self, input_dim_list):
        super(Multi_Cls, self).__init__()
        self.muti_fc = nn.ModuleList([
            nn.Linear(i, 20) for i in input_dim_list]
        )

    def forward(self, feature_list):
        classifier = [
            F.sigmoid(layer(feature).unsqueeze(1)) for feature, layer in zip(feature_list, self.muti_fc)
        ]
        return torch.cat(classifier, dim=1)


class MTS(nn.Module):
    def __init__(self, in_channels, in_dim, num_classes=1000):
        super(MTS, self).__init__()
        self.in_dim = in_dim
        self.norm = GlobalLayerNorm(num_features=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_dim, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        self.relu = nn.PReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(out_dim=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_dim=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_dim=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_dim=512, n_blocks=2, stride=2)
        self.avg_ada_pool = nn.AdaptiveAvgPool2d(1)
        self.max_ada_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self.input_dim_list = [128, 256, 512]
        self.mult_scale_mlp = MultiScaleFeatureFusion(input_dim_list=self.input_dim_list)
        self.muti_cls = Multi_Cls(self.input_dim_list)

    def _make_layer(self, out_dim, n_blocks, stride):
        layers = [Block(self.in_dim, out_dim, stride=stride)]
        # 先加入一个 stride 不为 1 的 block，对特征图进行下采样
        self.in_dim = out_dim
        # 再加入 stride 为 1 的若干 block，特征图大小保持不变
        for i in range(1, n_blocks):
            layers.append(Block(self.in_dim, out_dim, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        # stem
        x = self.norm(x)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.max_pool(x)
        pos = positional_encoding(64, 1).transpose(0, 1).view((1, 1, 64, 1)).to(x.device).float()
        x = x + pos
        # body
        x = self.layer1(x)
        x = self.layer2(x)

        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        avg_features = [self.avg_ada_pool(x).flatten(1) for x in features]
        # 为了连接全连接层 fc, 即 classifier ,需要将特征展成一维
        x1 = self.muti_cls(avg_features)
        x2 = self.mult_scale_mlp(avg_features)
        # x2 = torch.concat(avg_features,dim=1)
        # x2 = self.classifier(x2)
        #
        # return x2,x1,out_features
        # x2 = torch.cat(avg_features,dim=-1)
        # self.classifier(x2)
        return self.classifier(x2), x1


if __name__ == '__main__':
    x = torch.randn(16, 1, 257, 51)
    MTS = MTS(in_channels=1, in_dim=64, num_classes=10)
    y = MTS(x)
    print(y.shape)
