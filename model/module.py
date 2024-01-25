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


class Block(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            # nn.PReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm2d(output_channel)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x0):
        x1 = self.layer(x0)
        return self.relu(x0 + x1)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=64, output_nc=64, ngf=64, n_downsampling=2, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=8, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.Conv2d(input_nc, ngf, 9, stride=1, padding=4),
                  nn.BatchNorm2d(ngf),
                  nn.ReLU(inplace=True),
                 ]

        model2 = []
        # n_downsampling = n_downsampling
        # for i in range(n_downsampling):  # add downsampling layers
        #     mult = 2 ** i
        #     model2 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #               norm_layer(ngf * mult * 2),
        #               nn.ReLU(True)]

        # mult = 2 ** n_downsampling
        # for i in range(n_blocks):       # add ResNet blocks

        #     model2 += [ResidualBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # for i in range(n_downsampling):  # add upsampling layers
        #     mult = 2 ** (n_downsampling - i)
        #     model2 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
        #               norm_layer(int(ngf * mult / 2)),
        #               nn.ReLU(True)]
        # model2 += [nn.ReflectionPad2d(3)]
        # model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        # model2 += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        # self.model2 = nn.Sequential(*model2)

        self.model2 = nn.Sequential(
            Block(input_channel=ngf, output_channel=ngf),
            Block(input_channel=ngf, output_channel=ngf),
            Block(input_channel=ngf, output_channel=ngf),
            # Block(),
            # Block(),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        """Standard forward"""
        res = self.model1(input)
        return res + self.model3(self.model2(res))


class ResidualBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 1
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class shallowNet(nn.Module):
    def __init__(self, in_dim = 128, out_dim=1, n_downsampling=2, upscale=False):
        super(shallowNet, self).__init__()
        # if A2B:
        #       model = [nn.ReLU(True), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        # else:
        #   model = [nn.ReLU(True), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        if upscale:
            input = [nn.PReLU(), nn.ConvTranspose2d(in_dim, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        else:
            input = [nn.LeakyReLU(0.2, True), nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        self.resnet = ResnetGenerator(input_nc=64, output_nc=64, n_downsampling=n_downsampling, n_blocks=4)
        output = [nn.PReLU(), nn.Conv2d(64, out_dim, kernel_size=1, stride=1, padding=0, bias=False), nn.Tanh()]
        
        self.input = nn.Sequential(*input)
        self.output = nn.Sequential(*output)
    
    def forward(self, x):
        x = self.input(x)
        x = self.resnet(x)
        return self.output(x)
    





class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, B):
        """Standard forward"""
        return self.model(B)



class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] #+ up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
            return out
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
