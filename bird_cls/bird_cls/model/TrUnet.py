import torch
from torch import nn
from torch.nn import init

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
            nn.PReLU()
        )

    def forward(self, x):
        return self.StandardConv2d(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.DepthwiseSeparableConv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=stride,
                      padding=0),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.PReLU()
        self.cbam = CBAM(in_channels=out_channels, reduction_ratio=8)
        self.stride = stride
        self.bn_residu = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.residu_relu = nn.PReLU()

    def forward(self, x):
        out = self.DepthwiseSeparableConv2d(x)
        out1 = self.conv(x)
        out = self.relu(out + out1)
        atten = self.cbam(out)
        if self.stride != 1:
            out = self.bn_residu(x) + atten
        else:
            out = x + atten
        out = self.residu_relu(out)
        return out


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.DepthwiseSeparableConv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      groups=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True))
        self.cbam = CBAM(in_channels=out_channels, reduction_ratio=8)

    def forward(self, x):
        x = self.DepthwiseSeparableConv1d(x)
        x = self.cbam(x)
        return x


class StandardConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(StandardConv1d, self).__init__()
        self.StandardConv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=stride // 2),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.StandardConv1d(x)


class Identity(nn.Module):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x


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


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)
        if stride == 2 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down_sample = Identity()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        identity = self.down_sample(residual)
        x = self.relu(x + identity)
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_last, x):
        x = self.up(x_last) + x
        x = self.relu(self.norm(x))
        return x


class DwResNet(nn.Module):
    def __init__(self, in_channels, in_dim, num_classes=1000):
        super(DwResNet, self).__init__()
        self.in_dim = in_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_dim, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(out_dim=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(out_dim=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(out_dim=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(out_dim=512, n_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        # self.pcen = PCENTransform()

    def _make_layer(self, out_dim, n_blocks, stride):
        layers = [Block(self.in_dim, out_dim, stride=stride)]
        # 先加入一个 stride 不为 1 的 block，对特征图进行下采样
        self.in_dim = out_dim
        # 再加入 stride 为 1 的若干 block，特征图大小保持不变
        for i in range(1, n_blocks):
            layers.append(Block(self.in_dim, out_dim, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        # x = self.pcen(x[:, -1, ...].unsqueeze(dim=1))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # body
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 为了连接全连接层 fc, 即 classifier ,需要将特征展成一维
        x = self.avg_pool(x).flatten(1)
        fetures = x
        # x = F.dropout(x, p=0.7, training=self.training)
        x = self.classifier(x)
        return x,fetures


class SpatialAttention1D(nn.Module):
    def __init__(self):
        super(SpatialAttention1D, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class UpsampleAndConv(nn.Module):
    def __init__(self, in_channels, out_channels, target_height, target_width, kernel_size=1, stride=1, padding=0):
        super(UpsampleAndConv, self).__init__()
        self.upsample = nn.Upsample(size=(target_height, target_width), mode='bilinear', align_corners=False)
        self.cbam = CBAM(in_channels=out_channels, reduction_ratio=8)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x, x1):
        # 上采样
        upsampled_x = self.upsample(x)
        # 卷积调整通道大小
        output = self.conv(upsampled_x)
        output = self.bn(self.cbam(output) + x1)
        return output


class TrUnet(nn.Module):
    def __init__(self, in_channels=3, cls=2, ):
        super(TrUnet, self).__init__()
        self.in_channels = in_channels
        # log Mel 1 2 1 2 2 2
        # Log Spec 2 2 2 2 2 2
        self.down1 = DepthwiseSeparableConv2d(in_channels, 64, 3, 2)
        self.down2 = DepthwiseSeparableConv2d(64, 128, 3, 2)
        self.down3 = DepthwiseSeparableConv2d(128, 256, 3, 2)
        self.down4 = DepthwiseSeparableConv2d(256, 256, 3, 1)
        self.down5 = DepthwiseSeparableConv2d(256, 512, 3, 2)
        self.down6 = DepthwiseSeparableConv2d(512, 1024, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, cls)
        # self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        # self.up1 = UpsampleAndConv(in_channels=1024, out_channels=512, target_width=4, target_height=3, )
        # self.up_sample = nn.ConvTranspose2d(1024,out_channels=512,kernel_size=)

    def forward(self, x):
        # pcen_features = self.pcen(x[:, -1, ...].unsqueeze(dim=1))
        # pcen_features = (pcen_features - pcen_features.mean()) / pcen_features.std()
        # features = torch.cat((x[:, :-1, ...], pcen_features), dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        # x6 = self.up1(x5,x4)
        # x14 = self.cnn_1d(x_audio)
        # x9 = torch.concat([self.avgpool(x4).flatten(1), self.avgpool(x5).flatten(1), self.avgpool(x6).flatten(1), ],
        #                    dim=1)
        # x9 = torch.concat([self.avgpool(x6).flatten(1), ],
        #                   dim=1)
        # x10 = torch.concat(
        #     [self.maxpool(x5).flatten(1), self.maxpool(x6).flatten(1), self.maxpool(x3).flatten(1), ],
        #     dim=1)
        # x10 = torch.concat(
        #     [self.maxpool(x4).flatten(1), self.maxpool(x5).flatten(1), self.maxpool(x6).flatten(1),],
        #     dim=1)
        # x11 = torch.concat([x9.unsqueeze(1), x10.unsqueeze(1)], dim=1)
        # x11 = self.conv(x11).flatten(1)
        # x11 = F.relu(self.fc1(x11))
        x13 = self.fc2(self.avgpool(x6).flatten(1))
        return x13


if __name__ == '__main__':
    tr_unet = TrUnet(in_channels=1, cls=20)
    x = torch.randn(128, 1, 40, 51)
    y = tr_unet(x, None)
    pass
