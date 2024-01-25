 
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

import sys
sys.path.append("..")
from utils import *
from utils import set_requires_grad, weights_init_normal, ReplayBuffer, LambdaLR, save_sample, azimuthalAverage, plot_azimuthal
from metrics import eval, eval_6m
from model import UnetGenerator, FS_DiscriminatorA, FS_DiscriminatorB, NetworkA2B, NetworkB2A
from data import ImageDataset, ImageDataset_6mm
import metrics as ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL
from tqdm import tqdm
import os
from functools import lru_cache
from utils.func import Gaussian_pass, save_sample, tensor2img
import matplotlib.pyplot as plt
from PIL import Image
import baselines.cyclegan as cyclegan
import baselines.pseudo as pseudo


parser = argparse.ArgumentParser()
parser.add_argument('--n_downsampling', type=int, default=2, help='num of downsample')
parser.add_argument('--model_path', type=str, default='./checkpoint/netG_A2B_epoch31.pth', help='log path')
parser.add_argument('--l2h_hb', type=int, default=10, help='HF bandwidth for low2high resolution')
parser.add_argument('--l2h_lb', type=int, default=8, help='LF bandwidth for low2high resolution')
parser.add_argument('--h2l_hb', type=int, default=5, help='HF bandwidth for high2low resolution')
parser.add_argument('--h2l_lb', type=int, default=14, help='LF bandwidth for high2low resolution')
parser.add_argument('--sizeA', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--sizeB', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--method', type=str, default='ours', help='ours | cyclegan | pseudo')
opt = parser.parse_args()
print(opt)

@lru_cache()
def read_img(lr_path, hr_path, T_1, T_2):
    lr_img = Image.open(lr_path).convert('L')
    hr_img = Image.open(hr_path).convert('L')
    lr_img = T_1(lr_img).unsqueeze(0)
    hr_img = T_2(hr_img).unsqueeze(0)
    return lr_img, hr_img

lr = "./dataset/test/6mm_x2/"
hr = "./dataset/test/3mm/"
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

try:
    if opt.method == 'ours':
        netG_A2B = NetworkA2B(n_downsampling = opt.n_downsampling)
        model = torch.load(opt.model_path)
        netG_A2B.load_state_dict(model, strict=True)
        netG_A2B.eval()

        for i in tqdm(range(297)):
            lr_path = os.path.join(lr, str(i)+"_6.png")
            hr_path = os.path.join(hr, str(i)+"_3.png")
            if os.path.isfile(lr_path) and os.path.isfile(hr_path):
                
                lr_img, hr_img = read_img(lr_path, hr_path, T_1, T_2)
                
                hf = Gaussian_pass(lr_img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
                hf = (hf+lr_img)/2.0
                lf = Gaussian_pass(lr_img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
                _, _, sr_img = netG_A2B(lf, hf)
                output = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_sr.jpeg', output, cmap="gray")

                intput = lr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_lr.jpeg', intput, cmap="gray")

                gt = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_hr.jpeg', gt, cmap="gray")

    elif opt.method == 'cyclegan':
        netG_A2B = cyclegan.ResnetGenerator()
        model = torch.load('./baselines/baseline_output/baseline_output_cyclegan/netG_A2B_epoch100.pth')
        netG_A2B.load_state_dict(model, strict=True)
        netG_A2B.eval()

        for i in tqdm(range(297)):
            lr_path = os.path.join(lr, str(i)+"_6.png")
            hr_path = os.path.join(hr, str(i)+"_3.png")
            if os.path.isfile(lr_path) and os.path.isfile(hr_path):
                
                lr_img, hr_img = read_img(lr_path, hr_path, T_1, T_2)
                
                sr_img = netG_A2B(lr_img)
                output = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_sr.jpeg', output, cmap="gray")

                intput = lr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_lr.jpeg', intput, cmap="gray")

                gt = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_hr.jpeg', gt, cmap="gray")

    elif opt.method == 'pseudo':
        model_g_x2y = pseudo.RCAN(scale=1, num_rcab=5)
        model_sr = pseudo.RCAN(scale=2, num_rcab=10)
        model1 = torch.load('./baselines/baseline_output/baseline_output_pseudo/model_g_x2y55.pth')
        model2 = torch.load('./baselines/baseline_output/baseline_output_pseudo/model_sr55.pth')
        model_g_x2y.load_state_dict(model1, strict=True)
        model_sr.load_state_dict(model2, strict=True)
        model_g_x2y.eval()
        model_sr.eval()

        T_1 = transforms.Compose([ transforms.ToTensor(),
                    transforms.CenterCrop(opt.sizeA),
                    transforms.Resize([128, 128]),
                    transforms.Normalize((0.5), (0.5))
                    ])
        T_2 = transforms.Compose([ transforms.ToTensor(), 
                    transforms.CenterCrop(opt.sizeB),                        
                    transforms.Normalize((0.5), (0.5))])

        for i in tqdm(range(297)):
            lr_path = os.path.join(lr, str(i)+"_6.png")
            hr_path = os.path.join(hr, str(i)+"_3.png")
            if os.path.isfile(lr_path) and os.path.isfile(hr_path):
                
                lr_img, hr_img = read_img(lr_path, hr_path, T_1, T_2)
                
                sr_img = model_sr(model_g_x2y(lr_img))
                output = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_sr.jpeg', output, cmap="gray")

                intput = lr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_lr.jpeg', intput, cmap="gray")

                gt = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
                plt.imsave('./model_results/'+str(i)+'_hr.jpeg', gt, cmap="gray")
except ValueError:
    print("not available method")
        