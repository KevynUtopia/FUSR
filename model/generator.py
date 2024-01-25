from tkinter import NE

from cv2 import namedWindow
from pytorch_wavelets import DWTForward
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append("..")
from utils import high_pass, low_pass
from .module import *



class NetworkA2B(nn.Module):
    def __init__(self, n_downsampling=2, use_bias=False):
        super(NetworkA2B, self).__init__()
        self.unet = UnetGenerator(input_nc=64, output_nc=64, num_downs=7)
        # self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64),
        #                             nn.ReLU(True),nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(128),
        #                             nn.ReLU(True),nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64)
        #                               ])
        self.shallow_frequency  =   ResnetGenerator(input_nc=1, output_nc=64, n_downsampling=n_downsampling, n_blocks=1)               

        self.skip = nn.Sequential(*[nn.ReLU(True),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                        nn.BatchNorm2d(64)
                                      ])
    
        
        # self.A2B_input = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)
        #                               ])
        self.resnet = ResnetGenerator(input_nc=1, output_nc=64, n_blocks=5)
        self.shallow_up = shallowNet(upscale=False, n_downsampling=0)

    
    def forward(self, lf, hf):
        lf_feature = self.resnet(lf) #128^2
        hf_feature = self.resnet(hf) #64x128^2
        # zero_padding = torch.zeros_like(lf_feature) #64x256^2
        # hf_feature = self.deep(hf) #64x256^2
        # hf_feature = torch.zeros_like(hf_feature)

        cat_feature = torch.cat([lf_feature, hf_feature], 1)
        # return None, None, feature_map
        fuse_feature = self.shallow_up(cat_feature) #256^2
        return lf_feature, hf_feature, fuse_feature




class NetworkB2A(nn.Module):
    def __init__(self, n_downsampling=2, use_bias=False):
        super(NetworkB2A, self).__init__()

        self.shallow_frequency  =   ResnetGenerator(input_nc=1, output_nc=64, n_blocks=1)    
        
        # self.skip = nn.Sequential(*[nn.ReLU(True),
        #                                 nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #                                 nn.BatchNorm2d(64)
        #                               ])
        self.resnet = ResnetGenerator(input_nc=1, output_nc=64, n_blocks=5, n_downsampling=n_downsampling)
        self.B2A_input = nn.Sequential(*[nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)
                                      ])
        self.shallow_down = shallowNet(upscale=False, n_downsampling=0)


    
    def forward(self, hf, lf):
        hf_feature = self.resnet(hf) #64x256^2
        lf_feature = self.resnet(lf) #64x256^2
        # hf_feature = torch.zeros_like(hf_feature)

        cat_feature = torch.cat([hf_feature, lf_feature], 1)
        # return None, None, feature_map
        fuse_feature = self.shallow_down(cat_feature) #256^2
        return hf_feature, lf_feature, fuse_feature