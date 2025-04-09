import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TrUnet import CBAM


class Multi_feature(nn.Module):
    def __init__(self, in_channels, in_dim, num_heads, head_dim, block_size):
        super(Multi_feature, self).__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.query_dim = (in_dim - block_size) // block_size + 1
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=block_size, stride=block_size, bias=False,
                              out_channels=self.in_channels, groups=in_channels)
        # Linear layers for projecting input features into queries, keys, and values
        self.query_projection = nn.Parameter(torch.randn(self.query_dim, num_heads, head_dim))
        self.key_projection = nn.Parameter(torch.randn(self.query_dim, num_heads, head_dim))
        self.value_projection = nn.Parameter(torch.randn(self.query_dim, num_heads, head_dim))
        # Final linear layer for concatenating heads
        self.fc_out = nn.Linear(num_heads * head_dim, 512)
        self.fc1 = nn.Linear(512, 20)  # Assuming we have 10 class
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.3)
        self.norm = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        batch_size, input_channels, height, width = x.size()
        # Divide the input image into blocks
        x = self.conv(x)  # x(B C N T)
        # Project inputs to queries, keys, and values
        queries = torch.einsum('bciq,icj->bcj', x, self.query_projection)  # [batch_size, seq_len, num_heads * head_dim]
        keys = torch.einsum('bciq,icj->bcj', x, self.key_projection)  # [batch_size, seq_len, num_heads * head_dim]
        values = torch.einsum('bciq,icj->bcj', x, self.value_projection)  # [batch_size, seq_len, num_heads * head_dim]
        d_k = queries.shape[-1]
        # Compute scaled dot-product attention
        scaled_dot_product = torch.matmul(queries, keys.transpose(2, 1)) / math.sqrt(d_k)
        # Apply softmax along the last dimension
        attention_score = F.softmax(scaled_dot_product, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        # Compute output by multiplying attention weights with values
        attention_output = torch.matmul(attention_score, values)
        # fc
        x = self.fc_out(self.flatten(attention_output))
        features = x
        x = F.relu(self.dropout(x))
        x = self.fc1(x)
        return features, x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, enable_cbam=False):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.enable_cbam = enable_cbam
        if self.enable_cbam:
            self.cbam = CBAM(in_channels=out_channels, reduction_ratio=16)

    def forward(self, x):
        x = self.layer(x)
        if self.enable_cbam:
            x = self.cbam(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, cls):
        super(CNN, self).__init__()
        # Input channels = 3, output channels = 32
        self.norm = nn.BatchNorm2d(in_channels)
        self.layers = []
        self.cbam =[0,0,0,0,0]
        out_channels = 64
        for i in range(5):
            self.layers.append(Block(in_channels=in_channels, out_channels=out_channels,enable_cbam= self.cbam[i] is 1))
            in_channels = out_channels
            out_channels = out_channels * 2
        self.layers = nn.Sequential(*self.layers)
        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.cbam = CBAM(512, reduction_ratio=16)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, cls)  # Assuming we have 10 classes
        # self.pcen = PCENTransform()

    def forward(self, x):
        # x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        # Flatten the image
        x = self.avgpool(x).flatten(1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # To use the model
    x = torch.randn(1, 3, 257, 126)
    # import numpy as np
    # # 假设 Q^T 和 K^T 的形状都是 (1, 2, 3)，元素值分别为 1, 2, 3
    # Q_T = np.random.randn(1, 2,3)
    # K_T = np.random.randn(1, 2,3)
    #
    # # 假设 V 的形状也是 (1, 2, 3)，元素值也是 1, 2, 3
    # V = np.random.randn(1, 2,3)
    #
    # # 计算注意力分数矩阵
    # attention_scores = np.matmul(Q_T, K_T.transpose(0, 2, 1))
    #
    # # 将注意力分数矩阵与 V 进行加权求和
    # output = np.matmul(attention_scores, V)
    # pass
    model = Multi_feature(in_channels=3, in_dim=257, num_heads=3, head_dim=100, block_size=5)
    y = model(x)
    print(y.shape)
    # model = CNN(in_channels=1, cls=20)
    # y = model(x)
    # print(y.shape)
