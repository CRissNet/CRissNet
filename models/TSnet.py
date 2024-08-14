import torch
import torch.nn as nn
from collections import OrderedDict
from utils import logger
from torch.nn import Softmax

__all__ = ["TSnet"]


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class ConvBN1(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, groups, stride=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN1, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 10, 3)),
            ('relu1', nn.PReLU(num_parameters=10, init=0.3)),
            ('conv1x9', ConvBN(10, 10, [1, 9])),
            ('relu2', nn.PReLU(num_parameters=10, init=0.3)),
            ('conv9x1', ConvBN(10, 10, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 10, [5, 1])),
            ('relu', nn.PReLU(num_parameters=10, init=0.3)),
            ('conv5x1', ConvBN(10, 10, [1, 5])),
        ]))
        self.conv1x1 = ConvBN(10 * 2, 2, 1)
        self.identity = nn.Identity()  # 全连接层
        self.relu1 = nn.PReLU(num_parameters=20, init=0.3)
        self.relu2 = nn.PReLU(num_parameters=2, init=0.3)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)  # 0为行1为列
        out = self.relu1(out)
        out = self.conv1x1(out)

        out = self.relu2(out + identity)  # 深度残差网络
        return out


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize, width,
                                                              height,
                                                              height).permute(0,
                                                                              2,
                                                                              1,
                                                                              3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class Depthwise(nn.Module):
    def __init__(self, in_dim):
        super(Depthwise, self).__init__()
        self.conv2 = ConvBN1(in_planes=2, out_planes=2, kernel_size=[5, 1], groups=2)
        self.conv3 = ConvBN1(in_planes=2, out_planes=2, kernel_size=[1, 5], groups=2)
        self.conv4 = ConvBN(in_dim, 2, 1)

    def forward(self, x):
        # out = self.conv1(x)
        out = self.conv4(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out



class tsnet(nn.Module):
    def __init__(self, reduction=4):
        super(tsnet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        logger.info(f'reduction={reduction}')
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(in_channel, 4, 3)),
            ("relu1", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv1x9_bn", ConvBN(4, 4, [1, 9])),
            ("relu2", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv9x1_bn", ConvBN(4, 4, [9, 1])),
            ("relu3", nn.PReLU(num_parameters=4, init=0.3)),
        ]))
        self.attention = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(4, 48, 1)),
            ("relu1", nn.PReLU(num_parameters=48, init=0.3)),
            ("Criss-Cross", CrissCrossAttention(48)),
        ]))
        self.down = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(48, in_channel, 1)),
            ("relu1", nn.PReLU(num_parameters=in_channel, init=0.3)),
        ]))
        self.encoder2 = nn.Sequential(OrderedDict([
            ("DW", Depthwise(in_channel)),
            ("relu", nn.PReLU(num_parameters=in_channel, init=0.3)),
        ]))
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.PReLU(num_parameters=4, init=0.3)),
            ("conv1x1_bn", ConvBN(4, 2, 1)),
            ("relu2", nn.PReLU(num_parameters=2, init=0.3)),
        ]))
        self.convdown = nn.Conv2d(2, 2, 5, (1, 2), padding=2)
        self.encoder_fc = nn.Linear(total_size // 2, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)  # 论文中没提，但这里将2048压缩为512再解压缩
        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.PReLU(num_parameters=2, init=0.3)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n, c, h, w = x.detach().size()

        encode1 = self.encoder1(x)
        encode1 = self.attention(encode1)
        encode1 = self.down(encode1)
        encode1 = self.convdown(encode1)
        encode2 = self.encoder2(x)
        encode2 = self.convdown(encode2)
        out = torch.cat((encode1, encode2), dim=1)
        out = self.encoder_conv(out)
        out = self.encoder_fc(out.view(n, -1))

        out = self.decoder_fc(out).view(n, c, h, w)
        out = self.decoder_feature(out)
        out = self.sigmoid(out)

        return out


def TSnet(reduction):
    r""" Create a proposed CRNet.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of CRNet
    """

    model = tsnet(reduction=reduction)
    return model  # 最终返回整个模型的model（包括不同的层以及各层的初始化参数）