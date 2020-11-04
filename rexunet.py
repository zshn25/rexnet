"""
Implements ReXNet based U-Net 
Author: Zeeshan Khan Suri
Date: 04.11.2020
"""

import torch
from torch import nn
import torch.nn.functional as F

from rexnetv1 import ReXNetV1, LinearBottleneck, Swish

class ReXUNet(nn.Module):
	"""
	Implements ReXNet based U-Net 
	Author: Zeeshan Khan Suri
	Date: 04.11.2020
	"""
	def __init__(self, ):
	  super(ReXUNet, self).__init__()
	  rexnet = ReXNetV1().features
	  self.conv0 = rexnet[:4] # outputs 
	  self.conv1 = rexnet[4:5]
	  self.conv2 = rexnet[5:7]
	  self.conv3 = rexnet[7:9]
	  self.conv4 = rexnet[9:15]
	  self.conv5 = rexnet[15:]

	  self.decode_conv1 = LinearBottleneck(1280, 140, t=1, stride=1)
	  self.decode_conv2 = LinearBottleneck(140, 72, t=1, stride=1)
	  self.decode_conv3 = LinearBottleneck(72, 50, t=1, stride=1)
	  self.decode_conv4 = LinearBottleneck(50, 27, t=1, stride=1)
	  self.decode_conv5 = LinearBottleneck(27, 16, t=1, stride=1)
	  self.decode_conv6 = LinearBottleneck(16, 1, t=1, stride=1)

	  self.swish = Swish()
	  
	def forward(self, x): # x = (3,64,64)
	  x1 = self.conv0(x) # (16,32,32)
	  x2 = self.conv1(x1) # (27,16,16)
	  x3 = self.conv2(x2) # (50,8,8)
	  x4 = self.conv3(x3) # (72,4,4)
	  x5 = self.conv4(x4) # (140,2,2)
	  bottleneck = self.conv5(x5) # (1280,1,1)

	  decode1 = self.decode_conv1(bottleneck)
	  decode1 = F.interpolate(decode1, scale_factor=2, mode='bilinear', align_corners=False) # (140,2,2)
	  decode1 = decode1 + x5

	  decode2 = self.decode_conv2(decode1)
	  decode2 = F.interpolate(decode2, scale_factor=2, mode='bilinear', align_corners=False) # (72,4,4)
	  decode2 = decode2 + x4

	  decode3 = self.decode_conv3(decode2)
	  decode3 = F.interpolate(decode3, scale_factor=2, mode='bilinear', align_corners=False) # (50,8,8)
	  decode3 = decode3 + x3

	  decode4 = self.decode_conv4(decode3)
	  decode4 = F.interpolate(decode4, scale_factor=2, mode='bilinear', align_corners=False) # (27,16,16)
	  decode4 = decode4 + x2

	  decode5 = self.decode_conv5(decode4)
	  decode5 = F.interpolate(decode5, scale_factor=2, mode='bilinear', align_corners=False) # (16,32,32)
	  decode5 = decode5 + x1

	  decode6 = self.decode_conv6(decode5)
	  decode6 = F.interpolate(decode6, scale_factor=2, mode='bilinear', align_corners=False) # (1,64,64)

	  return [self.swish(decode6),
			  self.swish(decode5),
			  self.swish(decode4),
			  self.swish(decode3),
			  self.swish(decode2),
			  self.swish(decode1),]
