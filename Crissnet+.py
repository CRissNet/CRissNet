class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.a = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.c = nn.Parameter(torch.rand(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        # 64, 32(行width) 5 32(列height)  ——————  64*32(64*行)  -1  32(列)   ———————— 变为 64*32 32 -1 为了之后与key相乘(
        # 相当于transpose)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        # 64, 32(列height) 5 32(行width)  ——————  64*32(64*列)  -1  32(行)   ———————— 变为 64*32 32 -1 为了之后与key相乘(
        # 相当于transpose)
        # print(proj_query.shape)
        # print(proj_query_H.shape)
        # print(proj_query_W.shape)
        batch1, len1, channel1 = proj_query_H.size()
        # print(batch1, len1, channel1)
        proj_query_H1 = proj_query_H[:, 1:len1, :]
        proj_query_H2 = proj_query_H[:, 0:len1 - 1, :]
        z = proj_query_H[:, 0:1, :]
        padding = z - z
        # padding = torch.zeros(batch1, 1, channel1).to(device)
        # print(padding.shape)
        proj_query_H1 = torch.cat((proj_query_H1, padding), dim=1)
        proj_query_H2 = torch.cat((padding, proj_query_H2), dim=1)
        # print(proj_query_H1.shape, proj_query_H2.shape)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        # a = nn.Parameter(torch.rand(1))
        # b = nn.Parameter(torch.rand(1))
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(m_batchsize, width,
                                                              height,
                                                              height).permute(0,
                                                                              2,
                                                                              1,
                                                                              3)
        energy_H1 = (torch.bmm(proj_query_H1, proj_key_H)).view(m_batchsize, width,
                                                                height,
                                                                height).permute(0,
                                                                                2,
                                                                                1,
                                                                                3)
        energy_H2 = (torch.bmm(proj_query_H2, proj_key_H)).view(m_batchsize, width,
                                                                height,
                                                                height).permute(0,
                                                                                2,
                                                                                1,
                                                                                3)
        # print(energy_H.shape)
        # print(energy_H1.shape)
        # print(energy_H2.shape)
        energy_W = self.c * torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        # print(energy_W.shape)
        concate1 = self.softmax(torch.cat([energy_H, energy_H1, energy_H2, energy_W], 3))
        # print(concate1.shape)
        concate2 = self.softmax(torch.cat([energy_H1, energy_H2], 3))
        # print(concate2.shape)
        att_H = concate1[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_H1 = concate2[:, :, :, 0: height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                    height)
        att_H2 = concate2[:, :, :, height:2 * height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width,
                                                                                            height, height)
        att_W = concate1[:, :, :, height: height + width].contiguous().view(m_batchsize * height, width, width)
        # print(att_H.shape)
        # print(att_H1.shape)
        # print(att_H2.shape)
        # print(att_W.shape)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_H1 = torch.bmm(proj_value_H, att_H1.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,
                                                                                                               1)
        out_H2 = torch.bmm(proj_value_H, att_H2.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,
                                                                                                               1)
        # print(out_H.shape)
        # print(out_H1.shape)
        # print(out_H2.shape)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out_H = out_H + self.a * out_H1 + self.b * out_H2
        # print(out_H.shape)
        # print(out_W.shape)  # H为列注意力，W为行注意力
        return self.gamma * (out_H + out_W) + x