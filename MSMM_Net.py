from typing import List

import torch
from torch import nn, Tensor

from .MDST_MemCell import MDST_MemCell

__all__ = ["MSMM_Net"]


class MSMM_Net(nn.Module):

    def __init__(self, in_channels: int, hidden_channels_list: List[int], kernel_size_list: List[int],
                 forget_bias: float = 0.01):
        r"""
        :param in_channels:               输入帧的通道数
        :param hidden_channels_list:      每一个堆叠层的隐藏层通道数
        :param kernel_size_list:          每一个堆叠层的卷积核尺寸
        :param forget_bias:               偏移量
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels_list = hidden_channels_list
        self.layers = len(hidden_channels_list)
        self.forget_bias = forget_bias

        cell_list = nn.ModuleList([])
        conv_list = nn.ModuleList([])
        pool_list = nn.ModuleList([])
        upsample_list = nn.ModuleList([])
        for i in range(self.layers):
            input_channels = in_channels if i == 0 else hidden_channels_list[i - 1]
            cell_list.append(
                MDST_MemCell(in_channels=input_channels, hidden_channels=hidden_channels_list[i],
                             kernel_size=kernel_size_list[i], forget_bias=forget_bias)
            )
            if i + 1 < self.layers:
                pool_list.append(
                    nn.MaxPool2d(kernel_size=4, padding=1, stride=2)
                )
                conv_list.append(
                    nn.Conv2d(self.hidden_channels_list[i], self.hidden_channels_list[i + 1], kernel_size=4, padding=1,
                              stride=2)
                )
                upsample_list.append(
                    nn.ConvTranspose2d(self.hidden_channels_list[i + 1], self.hidden_channels_list[i], kernel_size=4,
                                       padding=1, stride=2)
                )

        self.cell_list = cell_list
        self.pool_list = pool_list
        self.conv_list = conv_list
        self.upsample_list = upsample_list

        self.conv_last = nn.Conv2d(in_channels=hidden_channels_list[0], out_channels=in_channels,
                                   kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=False)

    def forward(self, inputs: Tensor, out_len: int = 10) -> Tensor:
        r"""
        :param inputs:   输入序列
        :param out_len:  预测长度
        :return:         输出序列
        """
        device = inputs.device
        batch, sequence, channel, height, width = inputs.shape

        h = []  # 存储隐藏层
        c = []  # 存储cell记忆
        pred = []  # 存储预测结果

        # 初始化最开始的隐藏状态
        for i in range(self.layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_channels_list[i], height // (2 ** i), width // (2 ** i)).to(
                device)
            zero_tensor_c = torch.zeros(batch, self.hidden_channels_list[i], height // (2 ** i), width // (2 ** i)).to(
                device)
            h.append(zero_tensor_h)
            c.append(zero_tensor_c)

        m = torch.zeros(batch, self.hidden_channels_list[0], height, width).to(device)

        # 开始循环，模型在预测部分的输入是前一帧的预测输出
        for s in range(sequence + out_len):
            if s < sequence:
                x = inputs[:, s]
                if s == 0:
                    diff_x = x
                else:
                    diff_x = x - inputs[:, s - 1]
            else:
                diff_x = x_pred - x
                x = x_pred

            h[0], c[0], m, diff_x = self.cell_list[0](x, h[0], c[0], m, diff_x)
            x_t = []

            for i in range(1, self.layers):
                x_t.append(self.pool_list[i - 1](h[i - 1]))
                diff_x = self.pool_list[i - 1](diff_x)
                m = self.conv_list[i - 1](m)
                h[i], c[i], m, diff_x = self.cell_list[i](x_t[i - 1], h[i], c[i], m, diff_x)

            x_pred = self.upsample_list[-1](h[-1])
            for i in range(self.layers - 2, 0, -1):
                x_pred = self.upsample_list[i - 1](x_pred + h[i])

            x_pred = self.conv_last(x_pred + h[0])

            for i in range(self.layers - 2, -1, -1):
                m = self.upsample_list[i](m)

            if s >= sequence:
                pred.append(x_pred)

        # [length, batch, channel, height, width] -> [batch, length, channel, height, width]
        prediction = torch.stack(pred, dim=0).permute(1, 0, 2, 3, 4)

        return prediction
