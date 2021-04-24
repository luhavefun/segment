import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
import torchvision
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=4, stride=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x


class Trans_encoder_reshape(nn.Module):
    def __init__(self, patch_reso = [256,256],heads=8, dim=512,layers = 1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.Trans_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.Trans_encoder_layers = nn.TransformerEncoder(self.Trans_encoder_layer, num_layers=layers)
        self.patch_reso = patch_reso
        self.up_sample = nn.ConvTranspose2d(dim, dim//2, kernel_size=4, stride=4)
    def forward(self, x):
        B,L,C = x.shape
        H,W = self.patch_reso[0], self.patch_reso[1]
        x = self.Trans_encoder_layers(x)
        x = x.transpose(1,2)
        x = x.reshape(B,C,H,W)
        x = self.up_sample(x)
        return x
    

class Trans_block(nn.Module):
    def __init__(self, img_size=256,patch_size=4,dim=96,heads=8, layers=1):
        super().__init__()
        patch_reso = [img_size[0]//patch_size,img_size[1]//patch_size]
        self.Trans_embedding = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=dim)
        self.Trans_encoder = Trans_encoder_reshape(patch_reso, heads, dim, layers)
        
    def forward(self, x):
        x = self.Trans_embedding(x)
        x = self.Trans_encoder(x)
        
        return x


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = conv1x1(inplanes,planes,stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        
        return out
    
    
class BasicBlock(nn.Module):
    def __init__(self,input_size = [256,256],inplanes=3,planes=32,stride=2,patch_size=4,norm_layer=None,
                 heads=8, layers=1):
        super().__init__()
        self.Res_layer = Res_block(inplanes=inplanes, planes=planes, stride=stride,norm_layer=norm_layer)
        
        self.Transform = Trans_block([input_size[0]//2, input_size[1]//2], patch_size=patch_size, dim=2*planes,
                                     heads=heads,layers=layers)

    def forward(self, x):
        x = self.Res_layer(x)
        x = self.Transform(x)

        return x


class Up(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.reduce_dim = nn.Conv2d(dim,dim//2,3,1,padding=1)
        self.conv = nn.Conv2d(dim,dim//2, 3, 1, padding=1)
        
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x1 = self.reduce_dim(x1)
        x1 = torch.cat((x1, x2), dim=1)
        
        return self.conv(x1)
    
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_first = nn.Conv2d(3, 16, 3, 1, 1)
        self.layer1 = BasicBlock(input_size = [256,256],inplanes=16,planes=32,stride=2,patch_size=4,norm_layer=None,
                 heads=8, layers=1)
        self.layer2 = BasicBlock(input_size=[128, 128], inplanes=32, planes=64, stride=2, patch_size=4, norm_layer=None,
                                 heads=8, layers=1)
        self.layer3 = BasicBlock(input_size=[64, 64], inplanes=64, planes=128, stride=2, patch_size=4, norm_layer=None,
                                 heads=8, layers=1)
        self.layer4 = BasicBlock(input_size=[32, 32], inplanes=128, planes=256, stride=2, patch_size=4, norm_layer=None,
                                 heads=8, layers=1)
        self.maxpooling = nn.MaxPool2d(2,2)
        self.conv_mid = nn.Conv2d(256, 512, 3, 1, 1)
        self.Up_layer1 = Up(dim=512)
        self.Up_layer2 = Up(dim=256)
        self.Up_layer3 = Up(dim=128)
        self.Up_layer4 = Up(dim=64)
        self.Up_layer5 = Up(dim=32)

        self.conv_last = nn.Conv2d(16, 3, 3, 1, 1)
        
    def forward(self, x):
        x1 = self.conv_first(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.maxpooling(x5)
        x = self.conv_mid(x)
        x = self.Up_layer1(x, x5)
        x = self.Up_layer2(x, x4)
        x = self.Up_layer3(x, x3)
        x = self.Up_layer4(x, x2)
        x = self.Up_layer5(x, x1)
        x = self.conv_last(x)

        
        return x



# x = torch.rand(1,3,256,256)

# res = Res_block(inplanes=3, planes=32, stride=2)
# out = res(x)
# tran = Trans_block(img_size=[112,112], patch_size=4,dim=64,heads=8)
# out1 = tran(out)
# ba = BasicBlock(input_size = [224,224],inplanes=3,planes=32,stride=2,patch_size=4,norm_layer=None,
#                  heads=8, layers=1)
#
# out = ba(x)

# net = Net()
# out = net(x)
# y=1