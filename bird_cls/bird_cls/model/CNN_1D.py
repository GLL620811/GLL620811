import torch
from torch import nn

class CNN_1D(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """

    def __init__(self, L, N):
        super(CNN_1D, self).__init__()
        self.L, self.N = L, N
        # 50% overlap
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=L, stride=L // 2,bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),

        ) # N 256 L = (32000 - 160) / 80 +1 10ms  799
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=L, stride=L // 2, bias=False),
            nn.BatchNorm1d(512),
            nn.PReLU(),
        )  # N 256 L =
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=L, stride=L // 2, bias=False),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
        )
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=4, batch_first=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop_out = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(512, 20)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = self.conv1d_1(mixture)  # [M, N, K]
        mixture_w = self.conv1d_2(mixture_w)
        mixture_w = self.conv1d_3(mixture_w)
        mixture_w, _ = self.lstm(mixture_w.transpose(1, 2))
        mixture_w = mixture_w.transpose(1, 2)
        out = self.avgpool(mixture_w).flatten(1)
        out = F.dropout(out, p=0.5, training=self.training)
        f1 = out
        out = self.classifier(out)  # bs,1000
        return out
