import math
from torch import nn
import torch.nn.functional as F
import torch
from layers import *
from caplayer import *
from torch.nn import init
from thop import profile
from torchsummary import summary
import numpy as np
from thop import profile

def pixel_conv(feat, kernel):
    N, k2size, H, W = kernel.size()

    ksize = np.int(np.sqrt(k2size))
    pad   = (ksize-1)//2

    feat = F.pad(feat, (pad, pad, pad, pad))
    feat = feat.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat = feat.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat = feat.reshape(N, H, W, 1, -1)

    kernel = kernel.permute(0, 2, 3, 1).unsqueeze(-1)
    output = torch.matmul(feat, kernel)
    output = output.reshape(N, H, W, -1)
    output = output.permute(0, 3, 1, 2).contiguous()
    return output

class FSRCNN(nn.Module):#54kb 30.69 1.40 epo 95
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

class FireFSRCNN(nn.Module):#54kb 30.69 1.40 epo 95
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FireFSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [Fire(d, s, 6, 6)]
        for _ in range(m):
            self.mid_part.extend([Fire(s, 6, 6, 6)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

class ESPCN(nn.Module):#30.29 epo 92 1.04 98kb
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enco1 = nn.Conv2d(1,16,3,1,1)
        self.enco2 = Fire(16,4,16,16)
        self.enco3 = Fire(32,8,32,32)
        self.deco1 = Fire(64,16,64,64)
        self.deco2 = Fire(128,32,16,16)
        self.deco3 = Fire(32,8,8,8)
        self.pix_shu = nn.PixelShuffle(4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enco1(x))
        x = self.enco2(x)
        x = self.enco3(x)
        x = self.deco1(x)
        x = self.deco2(x)
        x = self.deco3(x)
        x = self.pix_shu(x)

        return x

class MobileSrv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.hs1 = nn.ReLU(inplace=True)
        self.bneck = nn.Sequential(
            Block(3, 16, 64, 64, nn.ReLU(inplace=True), SeModule(64), 1),
            Block(5, 64, 32, 32, nn.ReLU(inplace=True), SeModule(32), 1),
        )
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.hs2 = nn.ReLU(inplace=True)
        self.pix_shu = nn.PixelShuffle(4)
        self.hs3 = nn.ReLU(inplace=True)
        self.init_params()

    def init_params(self):
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
        out = self.hs1(self.conv1(x))
        out = self.bneck(out)
        out = self.hs2(self.conv2(out))
        out = self.hs3(self.pix_shu(out))


        return out

class MobileSrv1(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.PReLU()
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.PReLU(),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.PReLU(),
            )

        def conv_d(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.PReLU(),
            )

        def conv_p(inp,oup,stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.PReLU(),
            )


        self.model = nn.Sequential(
            conv_bn(1, 64, 1),
            #conv_bn(64, 64, 1),#190kb 30.58 1.07
            #EESP(64,64,1,4,7),#60kb 30.25 1.30
            #conv_d(64,64,1),#48kb 30.08 1.06
            #ContextGuidedBlock(64,64),#63kb 30.22 1.30
            #EESPnB(64,64,1,4,7),#55kb 30.25 1.24
            #nBContextGuidedBlock(64, 64),#60kb 30.19 1.24
            conv_bn(64, 16, 1),
        )
        self.pix_shu = nn.PixelShuffle(4)
        self.hs3 = nn.PReLU()

        self.upsam = nn.Sequential(
            nn.ConvTranspose2d(1,16,8,4,2),
            nn.PReLU(),
            conv_bn(16,1,1),
            nn.PReLU(),
        )


    def forward(self, x):
        up = self.upsam(x)
        out = self.model(x)
        out = self.pix_shu(out)
        out = self.hs3(out)

        return up+out

class Light10(nn.Module):
    def __init__(self):# 89kb 90epo:30.76 3.04
        super(Light10, self).__init__()
        self.head = nn.Conv2d(1, 64, 3, 1, 1)
        self.headprelu = nn.PReLU(64)
        self.eesp = EESPnB(64, 64, 1, 4, 7)
        self.eespcup1_2_1 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            eca_layer(128,k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup1_2_2 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            eca_layer(128, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup2_4_1 = nn.Sequential(
            EESPnB(32, 64, 1, 4, 7),
            eca_layer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.eespcup2_4_2 = nn.Sequential(
            EESPnB(64, 64, 1, 4, 7),
            eca_layer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.decoer = nn.Conv2d(32, 1, 3, 1, 1)
        self.decprelu = nn.PReLU(1)

    def forward(self, x):
        head = self.headprelu(self.head(x))
        up11 = self.eespcup1_2_1(head)
        x1 = self.eesp(head)
        up12 = self.eespcup1_2_2(x1)
        up12 = torch.cat((up11, up12), dim=1)
        up21 = self.eespcup2_4_1(up11)
        up22 = self.eespcup2_4_2(up12)
        up22 = torch.cat((up21, up22), dim=1)
        out = self.decoer(up22)
        out = self.decprelu(out)

        return out

class Light11(nn.Module):
    def __init__(self):
        #99kb 91epoch 30.86 3.14(4x 3ker 32-9*1)
        super(Light11, self).__init__()
        self.head = nn.Conv2d(1, 64, 3, 1, 1)
        self.headprelu = nn.PReLU(64)
        self.eesp = EESPnB(64, 64, 1, 4, 7)
        self.eespcup1_2_1 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            SELayer(128,k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup1_2_2 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            SELayer(128, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup2_4_1 = nn.Sequential(
            EESPnB(32, 64, 1, 4, 7),
            SELayer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.eespcup2_4_2 = nn.Sequential(
            EESPnB(64, 64, 1, 4, 7),
            SELayer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.decoer = nn.Conv2d(32, 1, 3, 1, 1)
        self.decprelu = nn.PReLU(1)

        self.kernels4x = nn.Sequential(
            nn.Conv2d(32, 9, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        head = self.headprelu(self.head(x))
        up11 = self.eespcup1_2_1(head)
        x1 = self.eesp(head)
        up12 = self.eespcup1_2_2(x1)
        up12 = torch.cat((up11, up12), dim=1)
        up21 = self.eespcup2_4_1(up11)
        up22 = self.eespcup2_4_2(up12)
        up22 = torch.cat((up21, up22), dim=1)
        out = self.decoer(up22)
        out = self.decprelu(out)
        z4 = self.kernels4x(up22)
        z4 = pixel_conv(out,z4)

        return out + z4

class Light12(nn.Module):
    def __init__(self):
        #inp, hidden_dim, oup, kernel_size, stride, use_se
        super(Light12, self).__init__()
        self.head = nn.Conv2d(1, 64, 3, 1, 1)
        self.headprelu = nn.PReLU(64)
        #self.eesp = EESPnB(64, 64, 1, 4, 7)
        self.eesp = nn.Sequential(
            GhostBottleneck(64, 64, 16, 1, 1, False),
            #nn.Conv2d(64,16,1,1,0),
            GhostBottleneck(16, 16, 16, 3, 1, False),
            #GhostBottleneck(16, 16, 64, 1, 1, True),
            nn.Conv2d(16, 64, kernel_size=1, padding=0),
            nn.PReLU(64),
        )
        self.eespcup1_2_1 = nn.Sequential(
            #EESPnB(64, 128, 1, 4, 7),
            #GhostBottleneck(64, 16, 16, 3, 1, False),
            #GhostBottleneck(16, 16, 16, 3, 1, False),
            GhostBottleneck(64, 64, 128, 3, 1, True),
            #nn.Conv2d(16, 128, kernel_size=1,padding=0),
            nn.PReLU(128),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup1_2_2 = nn.Sequential(
            #nn.Conv2d(64, 16, 1, 1, 0),
            #GhostBottleneck(64, 16, 16, 3, 1, False),
            #GhostBottleneck(16, 16, 16, 3, 1, False),
            GhostBottleneck(64, 64, 128, 3, 1, True),
            #nn.Conv2d(16, 128, kernel_size=1, padding=0),
            nn.PReLU(128),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup2_4_1 = nn.Sequential(
            #EESPnB(32, 64, 1, 4, 7),
            #GhostBottleneck(32, 16, 16, 3, 1, False),
            GhostBottleneck(32, 32, 64, 3, 1, True),
            #nn.Conv2d(16, 64, kernel_size=1, padding=0),
            nn.PReLU(64),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.eespcup2_4_2 = nn.Sequential(
            #GhostBottleneck(64, 16, 16, 3, 1, False),
            #GhostBottleneck(16, 16, 16, 3, 1, False),
            GhostBottleneck(64, 64, 64, 3, 1, True),
            #Fire(16, 2, 32, 32),
            #nn.Conv2d(16, 64, kernel_size=1, padding=0),
            #nn.PReLU(64),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.decoer = nn.Conv2d(32, 1, 3, 1, 1)
        self.decprelu = nn.PReLU(1)

        self.kernels4x = nn.Sequential(
            #nn.Conv2d(32, 9, 3, 1, 1),
            #nn.PReLU(9),
            Fire(32, 4, 4, 5),#13612
            #DPConv(32,9),#13874
            #nn.PRelu(9),
        )

    def forward(self, x):
        head = self.headprelu(self.head(x))
        up11 = self.eespcup1_2_1(head)
        x1 = self.eesp(head)
        up12 = self.eespcup1_2_2(x1)
        up12 = torch.cat((up11, up12), dim=1)
        up21 = self.eespcup2_4_1(up11)
        up22 = self.eespcup2_4_2(up12)
        up22 = torch.cat((up21, up22), dim=1)
        out = self.decoer(up22)
        out = self.decprelu(out)
        z4 = self.kernels4x(up22)
        z4 = pixel_conv(out,z4)

        return out + z4

class Light113(nn.Module):
    def __init__(self):
        #99kb 91epoch 30.86 3.14(4x 3ker 32-9*1)
        super(Light113, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.head = nn.Conv2d(3, 64, 3, 1, 1)
        self.headprelu = nn.PReLU(64)
        self.eesp = EESPnB(64, 64, 1, 4, 7)
        self.eespcup1_2_1 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            SELayer(128,k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup1_2_2 = nn.Sequential(
            EESPnB(64, 128, 1, 4, 7),
            SELayer(128, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(32),
        )
        self.eespcup2_4_1 = nn.Sequential(
            EESPnB(32, 64, 1, 4, 7),
            SELayer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.eespcup2_4_2 = nn.Sequential(
            EESPnB(64, 64, 1, 4, 7),
            SELayer(64, k_size=3),
            nn.PixelShuffle(2),
            nn.PReLU(16),
        )
        self.decoer = nn.Conv2d(32, 3, 3, 1, 1)
        self.decprelu = nn.PReLU(3)
        #self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        self.kernels4x = nn.Sequential(
            nn.Conv2d(32, 9, 3, 1, 1),
            nn.ReLU(),
        )
        self.respic = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        head = self.headprelu(self.head(x))
        up11 = self.eespcup1_2_1(head)
        x1 = self.eesp(head)
        up12 = self.eespcup1_2_2(x1)
        up12 = torch.cat((up11, up12), dim=1)
        up21 = self.eespcup2_4_1(up11)
        up22 = self.eespcup2_4_2(up12)
        up22 = torch.cat((up21, up22), dim=1)
        out = self.decoer(up22)
        out = self.decprelu(out)
        z4 = self.kernels4x(up22)
        out1 = self.respic(out)
        z4 = pixel_conv(out1,z4)

        return out + z4

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output,identity_data)
        return output

class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 32)

        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=256, out_channels=256*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1)

    def forward(self, data):
        out = self.conv1(data)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out

if __name__ == '__main__':
    import torch
    import os
    from thop import profile
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #model = FireFSRCNN(scale_factor=4)
    #model = EDSR()
    #model = ESPCN(4)
    model = Light11()
    model = model.cuda()
    #summary(model, input_size=(1, 6, 6))
    input = torch.randn(1, 1, 96, 96).cuda()
    flops, params = profile(model, inputs=(input,))
    print(flops)
    print(params)
    #a = torch.rand(4,1,6,6)
    #print(a.shape)
    #a = a.cuda()
    #b = model(a)
    #print(b.shape)