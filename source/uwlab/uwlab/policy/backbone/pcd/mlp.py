# Source: Jiayuan Gu https://github.com/Jiayuan-Gu/torkit3d
from torch import nn
from uwlab.policy.backbone.pcd.conv import Conv1dBNReLU
from uwlab.policy.backbone.pcd.linear import LinearBNReLU

__all__ = ["mlp_bn_relu", "mlp_relu", "mlp1d_bn_relu", "mlp1d_relu"]


def mlp_bn_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(LinearBNReLU(c_in, c_out, relu=True, bn=True))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(LinearBNReLU(c_in, c_out, relu=True, bn=False))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp1d_bn_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv1dBNReLU(c_in, c_out, 1, relu=True, bn=True))
        c_in = c_out
    return nn.Sequential(*layers)


def mlp1d_relu(in_channels, out_channels_list):
    c_in = in_channels
    layers = []
    for c_out in out_channels_list:
        layers.append(Conv1dBNReLU(c_in, c_out, 1, relu=True, bn=False))
        c_in = c_out
    return nn.Sequential(*layers)
