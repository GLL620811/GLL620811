
import os
import numpy as np
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score

# Define the SincConv1d layer
class SincConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate):
        super(SincConv1d, self).__init__()

        self.sample_rate = sample_rate
        self.kernel_size = kernel_size

        # Calculate the bandpass parameters
        low = 50.0
        high = self.sample_rate / 2 - 50.0
        low = torch.tensor(low / (self.sample_rate / 2), dtype=torch.float32)
        high = torch.tensor(high / (self.sample_rate / 2), dtype=torch.float32)

        # Design the bandpass filters
        n = torch.arange(1, self.kernel_size + 1, dtype=torch.float32)
        self.low_pass = nn.Parameter(torch.sin(2 * np.pi * low * n) / (np.pi * n))
        self.high_pass = nn.Parameter(torch.sin(2 * np.pi * high * n) / (np.pi * n))

        # Learnable parameters
        self.band_pass = nn.Parameter(torch.cos(2 * np.pi * (high + low) * n) / (np.pi * n))

        # Convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        # Update the convolutional layer weights
        self.conv.weight.data[0] = self.low_pass
        self.conv.weight.data[1] = -self.high_pass
        self.conv.weight.data[2] = self.band_pass

        return self.conv(x)


# Define the SincNet model
class NewSincNet(nn.Module):
    def __init__(self, num_classes, sample_rate):
        super(NewSincNet, self).__init__()
        self.sample_rate = sample_rate

        # SincConv layers with different kernel sizes
        self.sinc_conv1 = SincConv1d(1, 64, 251, sample_rate)
        self.sinc_conv2 = SincConv1d(1, 64, 511, sample_rate)
        self.sinc_conv3 = SincConv1d(1, 64, 1023, sample_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 10 classes for example

    def forward(self, x):
        x1 = torch.relu(self.sinc_conv1(x))
        x2 = torch.relu(self.sinc_conv2(x))
        x3 = torch.relu(self.sinc_conv3(x))

        # Global average pooling
        x1 = torch.mean(x1, dim=-1)
        x2 = torch.mean(x2, dim=-1)
        x3 = torch.mean(x3, dim=-1)

        # Concatenate features from different scales
        x = torch.cat([x1, x2, x3], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
