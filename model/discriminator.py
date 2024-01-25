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


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]#, nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    

class FS_DiscriminatorA(nn.Module):
    def __init__(self, input_nc=1, stride=1, kernel_size=5, n_layers=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='cat'):
        super(FS_DiscriminatorA, self).__init__()

        self.wgan = wgan
        self.n_input_channel = input_nc
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs
        

        self.net = Discriminator(input_nc=input_nc, n_layers=n_layers)
        # if cs=='sum':
        self.net_dwt_l = Discriminator(input_nc=1, n_layers=n_layers-1)
        # else:
        self.net_dwt_h = Discriminator(input_nc=3, n_layers=1)
        self.pool = nn.AvgPool2d(kernel_size=11, stride=11, padding=2)
        self.out_net = nn.Softmax()

    def forward(self, x, w_l=1.0, w_h=1.0, x_gt=None, y=None):
        dwt_l, dwt_h, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        if self.n_input_channel==2:
            x = self.net(torch.cat([ximg, x_gt], 1))
        else:
            x = self.net(ximg)
        # x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D_l = self.net_dwt_l(dwt_l)
        dwt_D_h = self.pool(self.net_dwt_h(dwt_h))
        # dwt_D_h = self.net_dwt_h(dwt_h)
        # dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return (torch.flatten(x_D))
        wl, wh = w_l/(w_l+w_h), w_h/(w_l+w_h)
        # return (torch.flatten(0.5*x + 0.5*wl*dwt_D_l + 0.5*wh*dwt_D_h))
        return (torch.flatten(0.6*x + 0.2*dwt_D_l + 0.2*dwt_D_h))
        # return (torch.flatten(x))

    def filter_wavelet(self, x, norm=False):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return LL, torch.sum(LH, HL, HH), x

        elif self.cs.lower() == 'each':
            return LL, LH, HL, HH, x
        elif self.cs.lower() == 'cat':
            return LL, torch.cat((LH, HL, HH), 1), x
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))


class FS_DiscriminatorB(nn.Module):
    def __init__(self, input_nc=1, stride=1, kernel_size=5, n_layers=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='cat'):
        super(FS_DiscriminatorB, self).__init__()

        self.wgan = wgan
        self.n_input_channel = input_nc
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs

        

        self.net = Discriminator(input_nc=input_nc, n_layers=n_layers)
        # if cs=='sum':
        self.net_dwt_l = Discriminator(input_nc=1, n_layers=n_layers-1)
        # else:
        self.net_dwt_h = Discriminator(input_nc=3, n_layers=1)
        self.pool = nn.AvgPool2d(kernel_size=11, stride=11, padding=2)
        self.out_net = nn.Softmax()


    def forward(self, x, w_l=1.0, w_h=1.0, x_gt=None, y=None):
        dwt_l, dwt_h, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        if self.n_input_channel==2:
            x = self.net(torch.cat([ximg, x_gt], 1))
        else:
            x = self.net(ximg)
        # x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D_l = self.net_dwt_l(dwt_l)
        dwt_D_h = self.pool(self.net_dwt_h(dwt_h))
        # dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return dwt_D
        wl, wh = w_l/(w_l+w_h), w_h/(w_l+w_h)
        # return (torch.flatten(0.5*x + 0.5*wl*dwt_D_l + 0.5*wh*dwt_D_h))
        return (torch.flatten(0.6*x + 0.2*dwt_D_l + 0.2*dwt_D_h))
        # return (torch.flatten(x))



    def filter_wavelet(self, x, norm=False):
        LL, Hc = self.DWT2(x)
        LH, HL, HH = Hc[0][:, :, 0, :, :], Hc[0][:, :, 1, :, :], Hc[0][:, :, 2, :, :]
        if norm:
            LH, HL, HH = LH * 0.5 + 0.5, HL * 0.5 + 0.5, HH * 0.5 + 0.5
        if self.cs.lower() == 'sum':
            return LL, torch.sum(LH, HL, HH), x

        elif self.cs.lower() == 'each':
            return LL, LH, HL, HH, x
        elif self.cs.lower() == 'cat':
            return LL, torch.cat((LH, HL, HH), 1), x
        else:
            raise NotImplementedError('Wavelet format [{:s}] not recognized'.format(self.cs))
