import glob
import random
import os
from PIL import Image
import numpy as np
import time
import datetime
import sys

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

# from tensorflow import summary
# from torch.utils.tensorboard import SummaryWriter
# import utils

import argparse
import itertools
import matplotlib.pyplot as plt

# from pytorch_wavelets import DWTForward

import pdb
import skimage.metrics

from tqdm import tqdm
import phasepack.phasecong as pc
import cv2
from scipy.interpolate import griddata

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_A=None, transforms_B=None, unaligned=False, mode='train'):
        self.transformA = transforms.Compose(transforms_A)
        self.transformB = transforms.Compose(transforms_B)

        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '6mm_x2') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '3mm') + '/*.*'))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('L')
        item_A = self.transformA(img_A)

        if self.unaligned:
            item_B = self.transformB(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('L'))
        else:
            item_B = self.transformB(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ImageDataset_6mm(Dataset):
    def __init__(self, root, transforms_A=None, transforms_B=None, mode='train'):
        self.transformA = transforms.Compose(transforms_A)
        self.transformB = transforms.Compose(transforms_B)


        self.files_A = sorted(glob.glob(os.path.join(root, 'LR') + '/*.*'))
        # self.files_B = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        path_B = path_A
        path_B = path_B.replace("_lr.", "_hr.").replace("LR", "HR")

        item_A = self.transformA(Image.open(path_A).convert('L'))
        item_B = self.transformB(Image.open(path_B).convert('L'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files_A)


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_sample(epoch, tensor, suffix="_real", Azimuthal=False):
    output = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    f = np.fft.fft2(output)
    fshift = np.fft.fftshift(f)
    amp = np.log(np.abs(fshift))
    plt.imsave('../checkpoint_baseline_cyclegan/image_'+str(epoch+1)+suffix+'.jpeg', output, cmap="gray")
    plt.imsave('../checkpoint_baseline_cyclegan/image_'+str(epoch+1)+suffix+'amplitude.jpeg', amp, cmap="gray")
    if Azimuthal:
        N = 200
        epsilon = 1e-7
        fshift += epsilon

        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(magnitude_spectrum)

        # Calculate the azimuthally averaged 1D power spectrum
        points = np.linspace(0,N,num=psd1D.size) # coordinates of a
        xi = np.linspace(0,N,num=N) # coordinates for interpolation
        lr = griddata(points,psd1D,xi,method='cubic')

        # interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
        # psd1D_total[cont,:] = interpolated  
        plt.plot(xi, lr)
        plt.savefig('../checkpoint_baseline_cyclegan/image_'+str(epoch+1)+suffix+'azimuthal.jpeg')
        plt.cla()


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc=1, output_nc=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        use_bias = False
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg

def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator

def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def FSIM(
    org_img: np.ndarray, pred_img: np.ndarray, T1: float = 0.85, T2: float = 160
) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.
    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.
    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.
    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """
    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    # for i in range(1):
        # Calculate the PC for original and predicted images
    pc1_2dim = pc(
        org_img[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
    )
    pc2_2dim = pc(
        pred_img[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
    )

    # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
    # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
    # calculate the sum of all these 6 arrays.
    pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
    pc2_2dim_sum = np.zeros(
        (pred_img.shape[0], pred_img.shape[1]), dtype=np.float64
    )
    for orientation in range(6):
        pc1_2dim_sum += pc1_2dim[4][orientation]
        pc2_2dim_sum += pc2_2dim[4][orientation]

    # Calculate GM for original and predicted images based on Scharr operator
    gm1 = _gradient_magnitude(org_img[:, :], cv2.CV_16U)
    gm2 = _gradient_magnitude(pred_img[:, :], cv2.CV_16U)

    # Calculate similarity measure for PC1 and PC2
    S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
    # Calculate similarity measure for GM1 and GM2
    S_g = _similarity_measure(gm1, gm2, T2)

    S_l = (S_pc ** alpha) * (S_g ** beta)

    numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def eval(model):
    lr = "../dataset/test/6mm_x2/"
    hr = "../dataset/test/3mm/"
    num, psnr, ssim, mse, nmi, fsim= 0, 0, 0, 0, 0, 0
    model.eval()
    T_1 = transforms.Compose([ transforms.ToTensor(),
                transforms.CenterCrop(size_A),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.2312), (0.1381)),
                 ])
    T_2 = transforms.Compose([ transforms.ToTensor(),
                transforms.CenterCrop(size_B),                         
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.2656), (0.1924)),
                ])
    for i in tqdm(range(297)):
        lr_path = os.path.join(lr, str(i)+"_6.png")
        hr_path = os.path.join(hr, str(i)+"_3.png")
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            lr_img = Image.open(lr_path).convert('L')
            hr_img = Image.open(hr_path).convert('L')

            lr_img = T_1(lr_img).cuda().unsqueeze(0)
            hr_img = T_2(hr_img).cuda().unsqueeze(0)

            sr_img = model(lr_img)

            yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            fsim += FSIM(gtimg, yimg)
            num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f FSIM: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num, fsim/num))

def train_eval(model):
    i = random.randint(0, dt_l - 1)
    img = dataset[i]['A']
    x = img.unsqueeze(0).cuda()
    # plt.imshow(img.squeeze(0), "gray")
    y = model(x)
    yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
    psnr = skimage.metrics.peak_signal_noise_ratio(yimg, img.squeeze(0).cpu().detach().numpy())
    ssim = skimage.metrics.structural_similarity(yimg, img.squeeze(0).cpu().detach().numpy())
    nmi = skimage.metrics.mean_squared_error(yimg, img.squeeze(0).cpu().detach().numpy())
    print("traning PSNR: %.4f SSIM: %.4f NMI: %.4f"%(psnr, ssim, nmi))

def eval_6m_baseline(model, dataset,):
    n = len(dataset)
    num, psnr, ssim, mse, nmi, fsim= 0, 0, 0, 0, 0, 0
    model.eval()
    for i in range(n):
        img = dataset[i]['A'].unsqueeze(0).cuda()
        gt = dataset[i]['B'].unsqueeze(0).cuda()


        y = model(img)

        yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
        
        gtimg = gt.cpu().detach().numpy().squeeze(0).squeeze(0)
        psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=2))
        ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
        mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
        nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
        fsim += FSIM(gtimg, yimg)
        num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f FSIM: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num, fsim/num))
    

# #### Defination of local variables
# input_nc = 1
# output_nc = 1
# batchSize = 1
# size_A, size_B = 256, 256
# lr = 2e-4
# n_epochs, epoch, decay_epoch = 100, 0, 10
# n_cpu = 2
# # dataroot = "./dataset/Colab_random_OCTA"
# dataroot = "../dataset/train"
# cuda = True



# ###### Definition of variables ######
# # Networks

# netG_A2B = ResnetGenerator()
# netG_B2A = ResnetGenerator()
# netD_A = Discriminator()
# netD_B = Discriminator()

# # netG_A2B = UnetGenerator()
# # netG_B2A = UnetGenerator()
# # netD_A = Discriminator()
# # netD_B = Discriminator()

# if cuda:
#     netG_A2B.cuda()
#     netG_B2A.cuda()
#     netD_A.cuda()
#     netD_B.cuda()

# netG_A2B.apply(weights_init_normal)
# netG_B2A.apply(weights_init_normal)
# netD_A.apply(weights_init_normal)
# netD_B.apply(weights_init_normal)

# # Lossess
# criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.L1Loss()
# # criterion_phase = phase_consistency_loss()
# criterion_identity = torch.nn.L1Loss()


# # Optimizers & LR schedulers
# optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(0.5, 0.999))


# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
# lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# # Inputs & targets memory allocation
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# input_A = Tensor(batchSize, input_nc, size_A, size_A)
# # input_B = Tensor(batchSize, output_nc, size_A, size_A)
# input_B = Tensor(batchSize, output_nc, size_B, size_B)
# target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

# fake_A_buffer = ReplayBuffer()
# fake_B_buffer = ReplayBuffer()

# # Dataset loader
# transforms_A = [ 
#                 transforms.ToTensor(),
#                 transforms.RandomCrop((size_A, size_A)),
#                 # transforms.Normalize((0.2627), (0.1739)),                
#                 transforms.Normalize((0.5), (0.5))                
#                 ]
                
# transforms_B = [ 
#                 transforms.ToTensor(),
#                 transforms.RandomCrop((size_B, size_B)),
#                 # transforms.Normalize((0.2642), (0.1943)),
#                 transforms.Normalize((0.5), (0.5)),
#                 ]
# dataset = ImageDataset(dataroot, transforms_A=transforms_A, transforms_B=transforms_B, unaligned=True)
# dt_l = len(dataset)
# print (len(dataset))
# dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)



# test_path = "../dataset/evalution_6mm/parts"
# transforms_A = [ 
#                 transforms.ToTensor(),                 
#                 transforms.CenterCrop(size_A),
#                 transforms.Normalize((0.5), (0.5)),
#                 # transforms.Resize((128, 128))
#                 ]
# transforms_B = [ 
#                 transforms.ToTensor(),
#                 transforms.CenterCrop(size_B),
#                 transforms.Normalize((0.5), (0.5))
#                 ]
# test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B)




# # Loss plot
# # logger = Logger(n_epochs, len(dataloader))
# ###################################
# netG_A2B.train()
# netG_B2A.train()
# netD_A.train()
# netD_B.train()

# globaliter = 0
# epoch = 0

# for epoch in range(epoch, n_epochs):
#     real_out, fake_out = None, None
#     for i, batch in enumerate(dataloader):
#         globaliter += 1
#         real_A = Variable(input_A.copy_(batch['A']))
#         real_B = Variable(input_B.copy_(batch['B']))

#         ######### (1) forward #########
#         fake_B = netG_A2B(real_A)
#         recovered_A = netG_B2A(fake_B)
#         fake_A = netG_B2A(real_B)
#         recovered_B = netG_A2B(fake_A)


#         ###### (2) G_A and G_B ######
#         set_requires_grad([netD_A, netD_B], False)
#         optimizer_G.zero_grad()

#         pred_fake = netD_B(fake_B)
#         loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

#         pred_fake = netD_A(fake_A)
#         loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

#         idt_A = netG_A2B(real_B)
#         loss_idt_B = criterion_identity(idt_A, real_B) * 0.5

#         idt_B = netG_B2A(real_A)
#         loss_idt_A = criterion_identity(idt_B, real_A) * 0.5
        

#         loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
#         loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

#         loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt_B + loss_idt_A

#         loss_G.backward()        
#         optimizer_G.step()

#         ###### (3) D_A and D_B ######
#         set_requires_grad([netD_A, netD_B], True)
#         optimizer_D.zero_grad()

#         # Real loss
#         pred_real = netD_A(real_A)
#         loss_D_real = criterion_GAN(pred_real, target_real)
#         # Fake loss
#         fake_A = fake_A_buffer.push_and_pop(fake_A)
#         pred_fake = netD_A(fake_A.detach())
#         loss_D_fake = criterion_GAN(pred_fake, target_fake)
#         # Total loss
#         loss_D_A = (loss_D_real + loss_D_fake)*0.5
#         loss_D_A.backward()


#         # Real loss
#         pred_real = netD_B(real_B)
#         loss_D_real = criterion_GAN(pred_real, target_real)      
#         # Fake loss
#         fake_B = fake_B_buffer.push_and_pop(fake_B)
#         pred_fake = netD_B(fake_B.detach())
#         loss_D_fake = criterion_GAN(pred_fake, target_fake)
#         # Total loss
#         loss_D_B = (loss_D_real + loss_D_fake)*0.5
#         loss_D_B.backward()

#         optimizer_D.step()

#         # writer.add_scalar('GAN A2B loss', loss_GAN_A2B, globaliter)
#         # writer.add_scalar('GAN B2A loss', loss_GAN_B2A, globaliter)
#         # writer.add_scalar('Cycle ABA loss', loss_cycle_ABA, globaliter)
#         # writer.add_scalar('Cycle BAB loss', loss_cycle_BAB, globaliter)
#         # writer.add_scalar('Identity A loss', loss_idt_A, globaliter)
#         # writer.add_scalar('Identity B loss', loss_idt_B, globaliter)
#         # writer.add_scalar('Dis A loss', loss_D_A, globaliter)
#         # writer.add_scalar('Dis B loss', loss_D_B, globaliter)
        
#         ####################################
#         ####################################

#         if i == 1:
#           input_tmp = Tensor(batchSize, input_nc, size_A, size_A)
#           x = Variable(input_tmp.copy_(batch['A']))
#           real_out = x
#           fake_out = netG_A2B(x)

#     # train_eval(netG_A2B)
#     eval(netG_A2B)
#     eval_6m_baseline(netG_A2B, test_dataset)
#     netG_A2B.train()

#     save_sample(epoch, real_out, "_input")
#     save_sample(epoch, fake_out, "_output", Azimuthal=True)

#     # Update learning rates
#     lr_scheduler_G.step()
#     lr_scheduler_D.step()


#     # Save models checkpoints
#     # torch.save(LR_encoding.state_dict(), 'output/LR_encoding.pth')
#     if epoch%5==4:
#       torch.save(netG_A2B.state_dict(), '../baseline_output_cyclegan/netG_A2B_epoch'+str(epoch+1)+'.pth')
#     print("Epoch (%d/%d) Finished" % (epoch+1, n_epochs))
