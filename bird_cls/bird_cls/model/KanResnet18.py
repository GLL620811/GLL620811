import torch
from torch import nn
from torch.nn import init
from model.kan_conv import *

class Identity(nn.Module):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride,device):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels,
            #           out_channels=out_channels,
            #           kernel_size=3,
            #           stride=stride,
            #           padding=1,
            #           bias=False,
            #           ),
            KAN_Convolutional_Layer(
                n_convs= out_channels// in_channels,
                kernel_size=(3, 3),
                device=device,
                padding=(1, 1),
                stride=(stride, stride),

            ),
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(in_channels=out_channels,
            #           out_channels=out_channels,
            #           kernel_size=3,
            #           stride=1,
            #           padding=1,
            #           bias=False,
            #           ),
            KAN_Convolutional_Layer(
                n_convs=1,
                kernel_size=(3, 3),
                device=device,
                padding=(1, 1),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = KAN_Convolutional_Layer(
                n_convs= 1,
                kernel_size=(3, 3),
                device=device,
                padding=(1, 1),
                stride=(1, 1),
            )
        if stride == 2 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                # KAN_Convolutional_Layer(
                #     n_convs= out_channels // in_channels,
                #     kernel_size=(1, 1),
                #     device=device,
                #     padding=(0, 0),
                #     stride=(stride, stride),
                # ),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False,),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down_sample = Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        identity = self.down_sample(residual)
        x = self.relu(x + identity)
        return x


class KanResnet18(nn.Module):
    def __init__(self, in_channels, in_dim, num_classes=1000,device='cpu'):
        super(KanResnet18, self).__init__()
        self.in_dim = in_dim
        self.conv1 = nn.Sequential(
            KAN_Convolutional_Layer(
                n_convs=in_dim,
                kernel_size=(7, 7),
                device=device,
                padding=(3,3),
                stride=(2,2),
            ),
            nn.BatchNorm2d(in_dim),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(out_dim=64, n_blocks=2, stride=1,device=device)
        self.layer2 = self._make_layer(out_dim=128, n_blocks=2, stride=2,device=device)
        self.layer3 = self._make_layer(out_dim=256, n_blocks=2, stride=2,device=device)
        self.layer4 = self._make_layer(out_dim=512, n_blocks=2, stride=2,device=device)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        # self.classifier = nn.
        # self.pcen = PCENTransform()

    def _make_layer(self, out_dim, n_blocks, stride,device):
        layers = [Block(self.in_dim, out_dim, stride=stride,device =device)]
        # 先加入一个 stride 不为 1 的 block，对特征图进行下采样
        self.in_dim = out_dim
        # 再加入 stride 为 1 的若干 block，特征图大小保持不变
        for i in range(1, n_blocks):
            layers.append(Block(self.in_dim, out_dim, stride=1,device = device))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        # x = self.pcen(x[:, -1, ...].unsqueeze(dim=1))
        x = self.conv1(x)
        x = self.max_pool(x)
        # body
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 为了连接全连接层 fc, 即 classifier ,需要将特征展成一维
        x = self.avg_pool(x).flatten(1)
        # x = F.dropout(x, p=0.7, training=self.training)
        x = self.classifier(x)
        return x
