import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class CAtten(nn.Module):
    def __init__(self, channel, out_channel):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, out_channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class EFBlock(nn.Module):
    '''
    Feature extraction ：特征提取模块
    in_channels:
    '''
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.c_attn = CAtten(channel=in_channels, out_channel=out_channels)
        self.s_attn = SpatialAttention(kernel_size=kernel_size)
        # self.atn = nn.Linear(out_channels, out_channels)
        self.fuse_conv = nn.Conv2d(out_channels*2, out_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        N, C, _, _ = x.size()
        fea_param_c = self.c_attn(y)
        # print("shape::::{}, N:{}, C:{}".format(fea_param_c.size(), N, C))
        # fea_param_c = self.atn(fea_param_c).reshape(N, C, 1, 1)
        out_c = x * fea_param_c
        out_s = x * self.s_attn(out_c)
        cond_feats = torch.cat((out_c, out_s), dim=1)
        cond_feats = self.fuse_conv(cond_feats)
        # 加入残差
        cond_feats += x
        cond_feats = self.relu(cond_feats)
        return cond_feats
