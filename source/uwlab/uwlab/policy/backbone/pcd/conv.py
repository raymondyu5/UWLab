# Source: Jiayuan Gu https://github.com/Jiayuan-Gu/torkit3d
from torch import nn

__all__ = ["Conv1dBNReLU", "Conv2dBNReLU"]


class Conv1dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
