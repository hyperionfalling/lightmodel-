import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from caplayer import *
import cv2

net = DPConv(32,1)

inputs = torch.rand(64,32,6,6)

f1= net(inputs)
print(f1.shape)





