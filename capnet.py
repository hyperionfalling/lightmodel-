import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from caplayer import *
from kpnlayer import *


class segSrCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, padding=2,bias=False),

        )
        self.step_1 = nn.Sequential(  # 1/2
            CapsuleLayer(1, 16, "conv", k=5, s=1, t_1=64, z_1=8, routing=1),
            CapsuleLayer(64, 8, "conv", k=5, s=1, t_1=1, z_1=25, routing=3),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.pix_shu = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.pix_shu(x)
        y = x
        x = self.conv_1(x)
        x.unsqueeze_(1)

        x = self.step_1(x)

        x.squeeze_(1)
        x = pixel_conv(y, x)
        return x + y
    def compute_vector_length(self, x):
        out = (x.pow(2)).sum(1, True)+1e-9
        out=out.sqrt()
        return out

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

class NoCaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1,bias=False),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0,bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),
            nn.ReLU(),
        )
        self.pix_shu = nn.PixelShuffle(4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder1(x)

        x = self.encoder2(x)

        x = self.encoder3(x)

        x = self.decoder1(x)

        x = self.decoder2(x)

        x = self.relu(self.pix_shu(x))

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

class KerSr(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.ReLU(),
        )
        self.pix_shu = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU()
        )

        self.kernels = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 25, 3, 1, 1, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        x = self.pix_shu(x)

        z = self.kernels(x)

        z = pixel_conv(x,z)

        return x + z




if __name__ == '__main__':
    import torch
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = SmallNet()
    #model = NoCaps()
    model = model.cuda()
    a = torch.ones(2, 1, 57, 6)
    print(a.shape)
    a = a.cuda()
    b = model(a)
    print(b.shape)