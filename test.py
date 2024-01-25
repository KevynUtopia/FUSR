import argparse
# import glob
# import random
# import os
# from PIL import Image
# import numpy as np
import time
# import datetime
# import sys
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchinfo import summary
import itertools
# import matplotlib.pyplot as plt
# import pdb
# import skimage.metrics
# from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import utils as func
from utils import set_requires_grad, weights_init_normal, ReplayBuffer, LambdaLR, save_sample, azimuthalAverage, plot_azimuthal
from metrics.metric import eval, eval_6m
from model.gan import NetworkA2B, NetworkB2A
from data.dataset import ImageDataset, ImageDataset_6mm
from tqdm import tqdm
import os
from functools import lru_cache
from utils import Gaussian_pass, save_sample, tensor2img
import matplotlib.pyplot as plt
from PIL import Image
import baselines.cyclegan as cyclegan
import baselines.pseudo as pseudo


parser = argparse.ArgumentParser()
parser.add_argument('--n_downsampling', type=int, default=2, help='num of downsample')
parser.add_argument('--model_path', type=str, default='./checkpoint/netG_A2B_35_ours.pth', help='log path') # './checkpoint/netG_A2B_35_ours.pth' ./checkpoint/netG_A2B_15_ours.pth
parser.add_argument('--l2h_hb', type=int, default=13, help='HF bandwidth for low2high resolution')
parser.add_argument('--l2h_lb', type=int, default=27, help='LF bandwidth for low2high resolution')
parser.add_argument('--h2l_hb', type=int, default=13, help='HF bandwidth for high2low resolution')
parser.add_argument('--h2l_lb', type=int, default=27, help='LF bandwidth for high2low resolution')
parser.add_argument('--sizeA', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--sizeB', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--method', type=str, default='ours', help='ours | cyclegan | pseudo')

parser.add_argument('--data_dir_lr', type=str, default="./dataset/test/6mm_x2/", help='ours | cyclegan | pseudo')
parser.add_argument('--data_dir_hr', type=str, default="./dataset/test/3mm/", help='ours | cyclegan | pseudo')
parser.add_argument('--data_dir_lr_6mm', type=str, default="./dataset/evaluation_6mm/whole/LR/", help='ours | cyclegan | pseudo')
parser.add_argument('--data_dir_hr_6mm', type=str, default="./dataset/evaluation_6mm/whole/HR/", help='ours | cyclegan | pseudo')
parser.add_argument('--whole_dir', type=str, default="./dataset/evaluation_6mm/parts", help='root directory of 6mm dataset')


parser.add_argument('--hfb', type=float, default=0.8, help='hfb weights')
parser.add_argument('--shallow_layer', type=int, default=3, help='# of layers of shallow extractor')
parser.add_argument('--deep_layer', type=int, default=10, help='# of layers of deep extractor')
opt = parser.parse_args()
print(opt)

@lru_cache()
def read_img(lr_path, hr_path, T_1, T_2):
    lr_img = Image.open(lr_path).convert('L')
    hr_img = Image.open(hr_path).convert('L')
    lr_img = T_1(lr_img).unsqueeze(0)
    hr_img = T_2(hr_img).unsqueeze(0)
    return lr_img, hr_img

lr = opt.data_dir_lr
hr = opt.data_dir_hr
lr_6mm = opt.data_dir_lr_6mm
hr_6mm = opt.data_dir_hr_6mm

T_1 = transforms.Compose([ transforms.ToTensor(),
            transforms.CenterCrop(opt.sizeA),
            # transforms.Resize((opt.sizeA, opt.sizeA)),
            transforms.Normalize((0.5), (0.5)),
            # transforms.Normalize((0.230), (0.138)),
                ])
T_2 = transforms.Compose([ transforms.ToTensor(),
            transforms.CenterCrop(opt.sizeB),                         
            transforms.Normalize((0.5), (0.5)),
            # transforms.Normalize((0.138), (0.193)),
            ])

test_path = opt.whole_dir
transforms_A = [ 
                transforms.ToTensor(),                 
                transforms.CenterCrop(opt.sizeA),
                # transforms.Resize((size_A, size_A)),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.280), (0.178)),
                ]
transforms_B = [ 
                transforms.ToTensor(),
                transforms.CenterCrop(opt.sizeB),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.285), (0.214))
                ]
test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B)
print(len(test_dataset))

try:
    if opt.method == 'ours':
        netG_A2B = NetworkA2B(n_downsampling = opt.n_downsampling, opt=opt).cuda()
        model = torch.load(opt.model_path)
        netG_A2B.load_state_dict(model, strict=True)
        netG_A2B.eval()

        psnr_3mm, ssim_3mm, mse_3mm, nmi_3mm, fsim_3mm, lfd_3mm = eval(netG_A2B, opt)
        psnr_6mm, ssim_6mm, mse_6mm, nmi_6mm, fsim_6mm, lfd_6mm = eval_6m(netG_A2B, test_dataset, opt)


except ValueError:
    print("not available method")


                