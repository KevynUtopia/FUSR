import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import lru_cache

# Filters
@lru_cache()
def   guais_low_pass(img, band=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols), dtype = float)
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = np.exp(-0.5 *  distance_u_v / (band ** 2))
    return torch.from_numpy(mask).float()

@lru_cache()
def   guais_high_pass(img, band=10):
    rows, cols = img.shape
    center = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), dtype = float)
    for i in range(rows):
        for j in range(cols):
            distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
            mask[i, j] = 1 - np.exp(-0.5 *  distance_u_v / (band ** 2))
    return torch.from_numpy(mask).float()

def high_pass(timg, high_pass = True, band=4):
    pass

@lru_cache()
def Gaussian_pass(timg, high_pass = True, band=13):
    lf = transforms.GaussianBlur(band)(timg)
    if high_pass:
        return ((timg-lf)).squeeze(0).squeeze(0)
    else:
        return (lf).squeeze(0).squeeze(0)

    # f = torch.fft.fft2(timg[0])
    # fshift = torch.fft.fftshift(f)

    # amp, pha = torch.abs(fshift), torch.angle(fshift)

    # if high_pass:
    #     mask = guais_high_pass(fshift, band).cuda()
    # else:
    #     mask = guais_low_pass(fshift, band).cuda()
    # amp = amp * mask

    # s1_real = amp*torch.cos(pha)
    # s1_imag = amp*torch.sin(pha)
    # s2 = torch.complex(s1_real, s1_imag)

    # ishift = torch.fft.ifftshift(s2)
    # iimg = torch.fft.ifft2(ishift)
    # iimg = torch.abs(iimg)
    # return iimg

def weighted_D(real, fake):
    f = torch.fft.fft2(real[0])
    fshift = torch.fft.fftshift(f)
    amp_real = torch.abs(fshift)

    
    f = torch.fft.fft2(fake[0])
    fshift = torch.fft.fftshift(f)
    amp_fake = torch.abs(fshift)
    
    hf_mask = guais_high_pass(fshift, band=20).cuda()
    lf_mask = guais_low_pass(fshift, band=20).cuda()

    hf_weight = torch.nn.MSELoss()(hf_mask*amp_real, hf_mask*amp_fake)
    lf_weight = torch.nn.MSELoss()(lf_mask*amp_real, lf_mask*amp_fake)
    # hf_weight = FFL(loss_weight=1.0, alpha=1.0)(hf_mask*amp_real.unsqueeze(0).unsqueeze(0), hf_mask*amp_fake.unsqueeze(0).unsqueeze(0)) 
    # lf_weight = FFL(loss_weight=1.0, alpha=1.0)(lf_mask*amp_real.unsqueeze(0).unsqueeze(0), lf_mask*amp_fake.unsqueeze(0).unsqueeze(0))
    return hf_weight, lf_weight

def low_pass(timg, band=10):
    pass

def bandreject_pass(timg, r_out=300, r_in=35):
    f = torch.fft.fft2(timg[0])
    fshift = torch.fft.fftshift(f)
    
    rows, cols = timg[0].shape
    crow,ccol = int(rows/2), int(cols/2)
    mask = bandreject_filters(fshift, r_out, r_in).cuda()
    
    f = fshift * mask
    
    ishift = torch.fft.ifftshift(f)
    iimg = torch.fft.ifft2(ishift)
    iimg = torch.abs(iimg)
    return iimg

def bandreject_filters(img, r_out=300, r_in=35):
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    radius_out = r_out
    radius_in = r_in

    mask = np.zeros((rows, cols))
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.logical_and(((x - center[0]) ** 2 + (y - center[1]) ** 2 >= r_in ** 2),
                               ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r_out ** 2))
    mask[mask_area] = 1
    return torch.from_numpy(mask).float()

def laplacian_kernel(im):
    conv_op = nn.Conv2d(1, 1, 3, bias=False, stride=1, padding=1)
    laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    laplacian_kernel = laplacian_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(laplacian_kernel).cuda()
    edge_detect = conv_op(Variable(im))
    return edge_detect

def functional_conv2d(im):
    sobel_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

