import torch
from torch import nn
from torch.nn import GroupNorm
from model.pool import PoolFormerBlock


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


class SGFEN(nn.Module):
    def __init__(self, input_channels, input_channels2):
        super(SGFEN, self).__init__()

        self.head = 4
        self.kernel_conv = 3
        self.kernel_att = 5
        self.dilation = 1
        self.stride = 1
        self.out_planes = 32
        self.head_dim = self.out_planes // self.head

        self.conv1h = conv_bn_relu(input_channels, 128, kernel_size=3, padding=1, bias=True)
        self.conv20 = conv_bn_relu(input_channels2, 8, kernel_size=3, padding=1, bias=True)
        self.conv10 = conv_bn_relu(8, 16, kernel_size=3, padding=1, bias=True)
        self.conv00 = conv_bn_relu(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv01 = conv_bn_relu(32, 64, kernel_size=3, padding=1, bias=True)

        self.conv1l = conv_bn_relu(64, 128, kernel_size=3, padding=1, bias=True)

        self.conv2h = conv_bn_relu(128, 64, kernel_size=3, padding=1, bias=True)
        self.conv2l = conv_bn_relu(128, 64, kernel_size=3, padding=1, bias=True)

        self.conv3h = conv_bn_relu(64, 32, kernel_size=3, padding=1, bias=True)

        self.f1 = conv_bn_relu(128, 32, kernel_size=1, padding=0, bias=True)
        self.f2 = conv_bn_relu(64, 32, kernel_size=1, padding=0, bias=True)
        self.f3 = conv_bn_relu(64, 32, kernel_size=1, padding=0, bias=True)

        """Parameters required by the cross-attention module"""

        self.conv1 = nn.Conv2d(32, self.out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        """conv4-conv7 are used in the subsequent feature fusion"""
        self.conv4 = conv_bn_relu(32, 32, kernel_size=1, padding=0)
        self.conv5 = conv_bn_relu(32, 32, kernel_size=1, padding=0)
        self.conv6 = conv_bn_relu(32, 32, kernel_size=1, padding=0)
        self.conv7 = conv_bn_relu(32, 32, kernel_size=1, padding=0)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transformer_decoder = PoolFormerBlock(
            dim=32, mlp_ratio=4., act_layer=nn.GELU, norm_layer=GroupNorm,
            drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5)

    def forward(self, x1, x2):
        x1 = self.conv1h(x1)

        x2 = self.conv20(x2)
        x2 = self.conv10(x2)
        x2 = self.conv00(x2)
        s = x2
        x2 = self.conv01(x2)

        x2 = self.conv1l(x2)

        y1 = x1 + x2
        y1 = self.f1(y1)

        x1 = self.conv2h(x1)
        x2 = self.conv2l(x2)

        y2 = x1 + x2
        y2 = self.f2(y2)

        x1 = self.conv3h(x1)

        x2 = self.transformer_decoder(s)

        scale = self.conv4((self.conv5(y1 + y2)))
        shift = self.conv6((self.conv7(y1 + y2)))
        x2 = x2 * scale + shift

        q, k, v = self.conv1(x2), self.conv2(x1), self.conv3(x1)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride
        pe = self.conv_p(position(h, w, x1.is_cuda))
        q_att = q.contiguous().view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.contiguous().view(b * self.head, self.head_dim, h, w)
        v_att = v.contiguous().view(b * self.head, self.head_dim, h, w)

        q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(
            b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(
            1, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)
        out_att = self.unfold(self.pad_att(v_att)).view(
            b * self.head, self.head_dim, self.kernel_att * self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
        x2 = torch.cat((x2, out_att), dim=1)
        x2 = self.f3(x2)

        x = x2
        X1 = self.avg_pool(x)
        X1 = torch.flatten(X1, 1)

        x = self.fc(X1)

        return x
