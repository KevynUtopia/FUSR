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
from utils import FilterHigh, FilterLow

from model.pose_hrnet import HighResolutionModule
from ..metrics.loss import phase_consistency_loss, PerceptualLoss

class Block(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            # nn.BatchNorm2d(output_channel),
            nn.PReLU(),

            nn.Conv2d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            # nn.BatchNorm2d(output_channel)
        )

    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=64, output_nc=64, ngf=64, n_downsampling=2, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=8):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model1 = [nn.Conv2d(input_nc, ngf, 9, stride=1, padding=4),
                #   nn.PReLU()
                nn.LeakyReLU(0.2)
                ]

        # model2 = []
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
        
        m2 = []
        for _ in range(n_blocks):
            m2 += [Block(input_channel=ngf, output_channel=ngf),]
            # m2 += [ResidualBlock(dim=ngf, norm_layer=norm_layer, use_dropout=use_dropout)]

        self.model2 = nn.Sequential( 
            *m2
            # Block(),
            # Block(),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(ngf),
        )

    def forward(self, input):
        """Standard forward"""
        res = self.model1(input)
        return self.model3(self.model2(res))


class ResidualBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_layer, use_dropout=False, use_bias=False):
        super(ResidualBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 1

        if not norm_layer:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.LeakyReLU(0.2)]
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.LeakyReLU(0.2)]
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


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
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]#nn.LeakyReLU(0.2, True)]
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

#################################################################
################# Frequency Discriminator ####################
#################################################################
class FS_DiscriminatorA(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, n_layers=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='cat'):
        super(FS_DiscriminatorA, self).__init__()

        self.wgan = wgan
        n_input_channel = 1
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs

        print('# FS type: {}, kernel size={}'.format(filter_type.lower(), kernel_size))
        

        self.net = Discriminator(input_nc=1, n_layers=n_layers)
        # if cs=='sum':
        self.net_dwt_l = Discriminator(input_nc=1, n_layers=n_layers-1)
        # else:
        self.net_dwt_h = Discriminator(input_nc=3, n_layers=1)
        self.pool = nn.AvgPool2d(kernel_size=11, stride=11, padding=2)
        self.out_net = nn.Softmax()

    def forward(self, x, w_l=1.0, w_h=1.0, y=None):
        dwt_l, dwt_h, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        # x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D_l = self.net_dwt_l(dwt_l)
        dwt_D_h = self.pool(self.net_dwt_h(dwt_h))
        # dwt_D_h = self.net_dwt_h(dwt_h)
        # dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return (torch.flatten(x_D))
        wl, wh = w_l/(w_l+w_h), w_h/(w_l+w_h)
        return (torch.flatten(0.5*x + 0.5*wl*dwt_D_l + 0.5*wh*dwt_D_h))
        # return (torch.flatten(*x))
        # return (torch.flatten(x + wl*dwt_D_l + wh*dwt_D_h))

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
    def __init__(self, recursions=1, stride=1, kernel_size=5, n_layers=5, wgan=False, highpass=True, D_arch='FSD',
                 norm_layer='Instance', filter_type='gau', cs='cat'):
        super(FS_DiscriminatorB, self).__init__()

        self.wgan = wgan
        n_input_channel = 1
        
        self.DWT2 = DWTForward(J=1, wave='haar', mode='reflect')
        self.filter = self.filter_wavelet
        self.cs = cs
        # n_input_channel = 3
        n_input_channel = 1

        print('# FS type: {}, kernel size={}'.format(filter_type.lower(), kernel_size))
        

        self.net = Discriminator(input_nc=1, n_layers=n_layers)
        # if cs=='sum':
        self.net_dwt_l = Discriminator(input_nc=1, n_layers=n_layers-1)
        # else:
        self.net_dwt_h = Discriminator(input_nc=3, n_layers=1)
        self.pool = nn.AvgPool2d(kernel_size=11, stride=11, padding=2)
        self.out_net = nn.Softmax()


    def forward(self, x, w_l=1.0, w_h=1.0, y=None):
        dwt_l, dwt_h, ximg = self.filter(x)
        # LL, LH, HL, HH, ximg = self.filter(x)
        x = self.net(ximg)
        # x_D = F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

        dwt_D_l = self.net_dwt_l(dwt_l)
        dwt_D_h = self.pool(self.net_dwt_h(dwt_h))
        # dwt_D = F.avg_pool2d(dwt_D, dwt_D.size()[2:]).view(x.size()[0], -1)

        # return dwt_D
        wl, wh = w_l/(w_l+w_h), w_h/(w_l+w_h)
        return (torch.flatten(0.5*x + 0.5*wl*dwt_D_l + 0.5*wh*dwt_D_h))
        # return (torch.flatten(x))
        # return (torch.flatten(x + wl*dwt_D_l + wh*dwt_D_h))
        # return (torch.flatten(x_D))



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




class NetworkA2B(nn.Module):
    def __init__(self, n_downsampling=2, use_bias=False):
        super(NetworkA2B, self).__init__()
        self.unet = UnetGenerator(input_nc=1, output_nc=64, num_downs=5)
        # self.shallow_frequency = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64),
        #                             nn.ReLU(True),nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(128),
        #                             nn.ReLU(True),nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),nn.BatchNorm2d(64)
        #                               ])
        self.shallow_frequency  =   ResnetGenerator(input_nc=1, output_nc=64, n_downsampling=n_downsampling, norm_layer=[], n_blocks=4)               

        self.skip = nn.Sequential(*[nn.ReLU(True),
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                        nn.BatchNorm2d(64)
                                      ])
    
        
        # self.A2B_input = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)
        #                               ])
        self.resnet = ResnetGenerator(input_nc=1, output_nc=64, norm_layer=[], n_blocks=10)
        self.shallow_up = shallowNet(upscale=False, n_downsampling=0)

    
    def forward(self, lf, hf):
        lf_feature = self.shallow_frequency(lf) #128^2
        hf_feature = self.resnet(hf) #64x128^2
        # zero_padding = torch.zeros_like(lf_feature) #64x256^2
        # hf_feature = self.deep(hf) #64x256^2
        
        cat_feature = torch.cat([lf_feature, hf_feature], 1)
        # return None, None, feature_map
        fuse_feature = self.shallow_up(cat_feature) #256^2
        return lf_feature, hf_feature, fuse_feature




class NetworkB2A(nn.Module):
    def __init__(self, n_downsampling=2, use_bias=False):
        super(NetworkB2A, self).__init__()
        self.unet = UnetGenerator(input_nc=1, output_nc=64, num_downs=4)
        self.shallow_frequency  =   ResnetGenerator(input_nc=1, output_nc=64, norm_layer=[], n_blocks=2)    
        
        # self.skip = nn.Sequential(*[nn.ReLU(True),
        #                                 nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #                                 nn.BatchNorm2d(64)
        #                               ])
        self.resnet = ResnetGenerator(input_nc=1, output_nc=64, norm_layer=[], n_blocks=5, n_downsampling=n_downsampling)
        self.B2A_input = nn.Sequential(*[nn.Conv2d(1, 128, kernel_size=4, stride=2, padding=1, bias=use_bias)
                                      ])
        self.shallow_down = shallowNet(upscale=False, n_downsampling=0)



    
    def forward(self, hf, lf):
        hf_feature = self.shallow_frequency(hf) #64x256^2
        lf_feature = self.resnet(lf) #64x256^2

        cat_feature = torch.cat([hf_feature, lf_feature], 1)
        # return None, None, feature_map
        fuse_feature = self.shallow_down(cat_feature) #256^2
        return hf_feature, lf_feature, fuse_feature




class shallowNet(nn.Module):
    def __init__(self, in_dim = 128, out_dim=1, n_downsampling=2, upscale=False):
        super(shallowNet, self).__init__()
        # if A2B:
        #       model = [nn.ReLU(True), nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        # else:
        #   model = [nn.ReLU(True), nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        # if upscale:
        #     input = [nn.PReLU(), nn.ConvTranspose2d(in_dim, 64, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64)]
        # else:
        #     input = [nn.PReLU(), nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64)]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        # model += [ResnetBlock()]
        input = [nn.PReLU(), nn.Conv2d(in_dim, 64, kernel_size=1, stride=1, padding=0, bias=False)]#, nn.BatchNorm2d(64)]
        self.resnet = ResnetGenerator(input_nc=64, output_nc=64, n_downsampling=n_downsampling, n_blocks=4)
        output = [nn.PReLU(), nn.Conv2d(64, out_dim, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU(0.2), nn.Tanh()]
        
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
            model = down + [submodule] + up
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
