"""
Implements ReXNet based U-Net 
Author: Zeeshan Khan Suri
Date: 04.11.2020

MIT License

Copyright (c) 2020 Zeeshan Khan Suri
"""

import torch
from torch import nn
import torch.nn.functional as F

from .rexnetv1 import ReXNetV1, LinearBottleneck, Swish

class ReXUNet(nn.Module):
    """
    Implements ReXNet based U-Net 
    Author: Zeeshan Khan Suri
    Date: 04.11.2020
    """
    def __init__(self, pretrained:str=None):
        """
        Inputs:
            pretrained: path to pretrained model
        """
        super(ReXUNet, self).__init__()
        rexnet = ReXNetV1()
        if pretrained:
            rexnet.load_state_dict(torch.load(pretrained))
        self.conv0 = rexnet.features[:4] # outputs 
        self.conv1 = rexnet.features[4:5]
        self.conv2 = rexnet.features[5:7]
        self.conv3 = rexnet.features[7:9]
        self.conv4 = rexnet.features[9:15]
        self.conv5 = rexnet.features[15:-1]
        
        self.decode_conv1 = LinearBottleneck(1280, 140, t=1, stride=1)
        self.decode_conv2 = LinearBottleneck(140, 72, t=1, stride=1)
        self.decode_conv3 = LinearBottleneck(72, 50, t=1, stride=1)
        self.decode_conv4 = LinearBottleneck(50, 27, t=1, stride=1)
        self.decode_conv5 = LinearBottleneck(27, 16, t=1, stride=1)
        self.decode_conv6 = LinearBottleneck(16, 1, t=1, stride=1)

        self.output_conv1 = nn.Conv2d( 16, 1, 3, 1, 1)
        self.output_conv2 = nn.Conv2d( 27, 1, 3, 1, 1)
        self.output_conv3 = nn.Conv2d( 50, 1, 3, 1, 1)
        self.output_conv4 = nn.Conv2d( 72, 1, 3, 1, 1)
        self.output_conv5 = nn.Conv2d(140, 1, 3, 1, 1)                            
        
        self.op_nonlin = nn.Sigmoid() #Swish(inplace=True)
        
    def forward(self, x): # x = (3,64,64)
        x1 = self.conv0(x) # (16,32,32)
        x2 = self.conv1(x1) # (27,16,16)
        x3 = self.conv2(x2) # (50,8,8)
        x4 = self.conv3(x3) # (72,4,4)
        x5 = self.conv4(x4) # (140,2,2)
        bottleneck = self.conv5(x5) # (1280,2,2)

        decode1 = self.decode_conv1(bottleneck)  # (140,2,2)
        decode1 = decode1 + x5
        output5 = self.op_nonlin(self.output_conv5(decode1))
        
        decode2 = self.decode_conv2(decode1)
        decode2 = F.interpolate(decode2, scale_factor=2, mode='bilinear', align_corners=False) # (72,4,4)
        decode2 = decode2 + x4
        output4 = self.op_nonlin(self.output_conv4(decode2))

        decode3 = self.decode_conv3(decode2)
        decode3 = F.interpolate(decode3, scale_factor=2, mode='bilinear', align_corners=False) # (50,8,8)
        decode3 = decode3 + x3
        output3 = self.op_nonlin(self.output_conv3(decode3))

        decode4 = self.decode_conv4(decode3)
        decode4 = F.interpolate(decode4, scale_factor=2, mode='bilinear', align_corners=False) # (27,16,16)
        decode4 = decode4 + x2
        output2 = self.op_nonlin(self.output_conv2(decode4))

        decode5 = self.decode_conv5(decode4)
        decode5 = F.interpolate(decode5, scale_factor=2, mode='bilinear', align_corners=False) # (16,32,32)
        decode5 = decode5 + x1
        output1 = self.op_nonlin(self.output_conv1(decode5))

        output0 = self.decode_conv6(decode5)
        output0 = F.interpolate(output0, scale_factor=2, mode='bilinear', align_corners=False) # (1,64,64)
        output0 = self.op_nonlin(output0)
        
        return [output0, output1, output2, output3, output4, output5]
    
class ReXDepth(ReXUNet):
    def __init__(self, scales=[0,1,2,3], pretrained_path:str=None):
        super(ReXDepth, self).__init__(pretrained=pretrained_path)
        self.scales = scales
    def forward(self,x):
        out = {}
        [out["disp0"], out["disp1"], 
                    out["disp2"], out["disp3"], 
                    out["disp4"], out["disp5"]] = super().forward(x)
        output = {}
        for scale in self.scales:
            output[("disp", scale)] = out["disp{}".format(scale)]

        return output
