from typing import Tuple

import torch
from torch import nn, Tensor

from .MDAM import MDAM

__all__ = ["MDST_MemCell"]


class MDST_MemCell(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1))

        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.mdam = MDAM(in_channel=hidden_channels)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, stride=1,
                              padding=padding)

    def forward(self, x: Tensor, h: Tensor, c: Tensor, m: Tensor, diffx: Tensor) -> Tuple[
        Tensor, Tensor, Tensor, Tensor]:
        r"""
        :param x:   输入图像，shape 为 (B, in_channels, H, W)
        :param h:   时间方向隐藏状态，shape 为 (B, hidden_channels, H, W)
        :param c:   cell记忆，shape 为 (B, hidden_channels, H, W)
        :param m:   空间方向隐藏状态，shape 为 (B, hidden_channels, H, W)
        :param diffx: 帧差，shape 为 (B, in_channels, H, W)
        :return:    h, c, m, diffx
        """
        if x is None and (h is None or c is None or m is None):
            raise ValueError("x 和 [h, c, m] 不能同时为 None")

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, f_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)

        c = f * c + i * g

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m)

        m = ff * m + ii * gg

        states = torch.cat([c, m], dim=1)

        o = torch.sigmoid(o_x + o_h + self.conv_o(states))
        h = o * torch.tanh(self.conv1x1(states))

        h, diffx = self.mdam(h, diffx)

        return h, c, m, diffx
