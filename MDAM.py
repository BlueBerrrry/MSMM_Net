import torch
from torch import nn


class MDAM(nn.Module):
    def __init__(self, in_channel, reduction=16, spatial_kernel=7):
        super(MDAM, self).__init__()

        if in_channel < 16:
            self.in_channel = 16
        else:
            self.in_channel = in_channel

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel // reduction, self.in_channel, 1, bias=False)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_fusion = nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.in_channel, kernel_size=1)
        self.conv_z2h = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel * 3, kernel_size=5, stride=1, padding=2,
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.in_channel * 3)
        )
        self.conv_h2h = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=3 * self.in_channel, kernel_size=5, stride=1, padding=2,
                      bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.in_channel * 3)
        )
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=self.in_channel, kernel_size=1)
        self.last_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=1, kernel_size=1)

    def forward(self, h, diffx):

        diffx = self.first_conv(diffx)
        _, h_channel, _, _ = h.size()
        if h_channel == 1:
            h = self.first_conv(h)

        h1 = self.max_pool(h)
        h2 = self.avg_pool(h)
        h_max_out = self.mlp(h1)
        h_avg_out = self.mlp(h2)
        h_channel_out = self.sigmoid(h_max_out + h_avg_out)
        h = h_channel_out * h

        h_max_out, _ = torch.max(h, dim=1, keepdim=True)
        h_avg_out = torch.mean(h, dim=1, keepdim=True)
        h_spatial_out = self.sigmoid(self.conv(torch.cat([h_max_out, h_avg_out], dim=1)))
        h = h_spatial_out * h

        diffx1 = self.max_pool(diffx)
        diffx2 = self.avg_pool(diffx)
        d_max_out = self.mlp(diffx1)
        d_avg_out = self.mlp(diffx2)
        d_channel_out = self.sigmoid(d_max_out + d_avg_out)
        diffx = d_channel_out * diffx

        d_max_out, _ = torch.max(diffx, dim=1, keepdim=True)
        d_avg_out = torch.mean(diffx, dim=1, keepdim=True)
        d_spatial_out = self.sigmoid(self.conv(torch.cat([d_max_out, d_avg_out], dim=1)))
        diffx = d_spatial_out * diffx

        # fusion
        z = self.conv_fusion(torch.cat([h, diffx], dim=1))

        h2h = self.conv_z2h(h)
        z2h = self.conv_h2h(z)

        (i, g, o) = torch.split(h2h + z2h, self.in_channel, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        diff_out = i * g + (1 - i) * diffx
        h_out = o * diff_out

        if h_channel == 1:
            h_out = self.last_conv(h_out)
        diff_out = self.last_conv(diff_out)

        return h_out, diff_out
