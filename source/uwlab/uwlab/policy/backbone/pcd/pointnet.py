import torch
import torch.nn as nn
from uwlab.policy.backbone.pcd.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu


class PointNet(nn.Module):
    """PointNet encoder. Input: (B, 3, N). Output: (B, global_channels[-1])."""

    def __init__(
        self,
        in_channels: int = 3,
        local_channels: tuple = (64, 64, 64, 128, 1024),
        global_channels: tuple = (512, 256),
        use_bn: bool = False,
    ):
        super().__init__()
        self.out_channels = (local_channels + global_channels)[-1]

        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm1d):
                module.momentum = 0.01

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, 3, N)
        local_feature = self.mlp_local(points)             # (B, C, N)
        global_feature, _ = torch.max(local_feature, dim=2)  # (B, C)
        return self.mlp_global(global_feature)             # (B, out_channels)
