#!/usr/bin/python3


import logging
import argparse
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
from torchinfo import summary
import itertools
import matplotlib.pyplot as plt
import pdb
import skimage.metrics
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import utils
from utils import set_requires_grad, weights_init_normal, ReplayBuffer, LambdaLR, save_sample, azimuthalAverage
from metric import eval, eval_6m
from model.gan import UnetGenerator, FS_DiscriminatorA, FS_DiscriminatorB, phase_consistency_loss, NetworkA2B, NetworkB2A, PerceptualLoss, TVLoss
from dataset import ImageDataset, ImageDataset_6mm
import ssim

# from tensorflow import summary
# from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default="./dataset/train", help='root directory of the dataset')
parser.add_argument('--whole_dir', type=str, default="./dataset/evalution_6mm/parts", help='root directory of 6mm dataset')
parser.add_argument('--pretrained_root', type=str, default="./pre_trained/netG_A2B_pretrained.pth", help='root directory of the pre-trained model')
parser.add_argument('--pretrained', type=bool, default=False, help='whether use pre-trained model')


parser.add_argument('--log_dir_train', type=str, default='./logs/tensorboard/train/', help='log path')
parser.add_argument('--log_dir_test', type=str, default='./logs/tensorboard/test/', help='log path')
parser.add_argument('--log', type=str, default='0', help='log file index')

parser.add_argument('--B2A', type=bool, default=False, help='save netB2A')
parser.add_argument('--scheduler', type=bool, default=True, help='save netB2A')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--sizeA', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--sizeB', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--n_layers', type=int, default=5, help='layer of the discriminators')
parser.add_argument('--dis_size', type=int, default=6, help='output size of discriminators')
parser.add_argument('--n_downsampling', type=int, default=2, help='num of downsample')

parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')

parser.add_argument('--seed', type=int, default=3074, help='Seed')
parser.add_argument('--beta1', type=float, default=0.25, help='beta 1')
parser.add_argument('--beta2', type=float, default=10.0, help='beta 2')
parser.add_argument('--beta3', type=float, default=2.0, help='beta 3')
parser.add_argument('--beta4', type=float, default=0.5, help='beta 4')
parser.add_argument('--beta5', type=float, default=0.5, help='beta 5')

parser.add_argument('--l2h_hb', type=int, default=10, help='HF bandwidth for low2high resolution')
parser.add_argument('--l2h_lb', type=int, default=8, help='LF bandwidth for low2high resolution')
parser.add_argument('--h2l_hb', type=int, default=5, help='HF bandwidth for high2low resolution')
parser.add_argument('--h2l_lb', type=int, default=14, help='LF bandwidth for high2low resolution')
opt = parser.parse_args()
print(opt)

SEED = opt.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

input_nc = opt.input_nc
output_nc = opt.output_nc
batchSize = opt.batchSize
size_A, size_B = opt.sizeA, opt.sizeB
lr = opt.lr
n_epochs, epoch, decay_epoch = opt.n_epochs, opt.epoch, opt.decay_epoch
n_cpu = opt.n_cpu
dataroot = opt.dataroot
cuda = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = NetworkA2B(n_downsampling = opt.n_downsampling)
netG_B2A = NetworkB2A(n_downsampling = opt.n_downsampling)
netD_A = FS_DiscriminatorA(input_nc, n_layers=opt.n_layers)
netD_B = FS_DiscriminatorB(output_nc, n_layers=opt.n_layers)

if cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

if not opt.pretrained:
    netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

summary(netG_A2B)
summary(netD_A)

# Lossess
criterion_GAN = torch.nn.MSELoss()
# criterion_cycle = torch.nn.SmoothL1Loss()
criterion_cycle = torch.nn.L1Loss()
criterion_phase = phase_consistency_loss()
criterion_identity = torch.nn.L1Loss()
# criterion_perceptual = PerceptualLoss(torch.nn.MSELoss())
criterion_ssim = ssim.SSIM()
criterion_ssim_TV_loss= TVLoss().cuda()
criterion_feature = torch.nn.BCEWithLogitsLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(0.9, 0.999))

if opt.scheduler:
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)
else:
    lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=0, verbose=True)
    lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=n_epochs, eta_min=0, verbose=True)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batchSize, input_nc, size_A, size_A)
# input_B = Tensor(batchSize, output_nc, size_A, size_A)
input_B = Tensor(batchSize, output_nc, size_B, size_B)
target_real = Variable(Tensor(opt.dis_size, opt.dis_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.dis_size, opt.dis_size).fill_(0.0), requires_grad=False)

target_real = torch.flatten(target_real)
target_fake = torch.flatten(target_fake)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

###################################
# Dataset loader
###################################
# load training set
transforms_A = [ 
                transforms.ToTensor(),
                transforms.RandomCrop((size_A, size_A)),
                # transforms.Normalize((0.2627), (0.1739)),                
                transforms.Normalize((0.5), (0.5))                
                ]
                
transforms_B = [ 
                transforms.ToTensor(),
                transforms.RandomCrop((size_B, size_B)),
                # transforms.Normalize((0.2642), (0.1943)),
                transforms.Normalize((0.5), (0.5)),
                ]

dataset = ImageDataset(dataroot, transforms_A=transforms_A, transforms_B=transforms_B, unaligned=True)
print (len(dataset))


dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

###################################
# load 3mm eval set
# hard-coded in util
pass


###################################
# load (whole) 6mm eval set

test_path = opt.whole_dir
transforms_A = [ 
                transforms.ToTensor(),                 
                transforms.CenterCrop(size_A*2),
                transforms.Resize((size_A, size_A)),
                transforms.Normalize((0.5), (0.5)),
                ]
transforms_B = [ 
                transforms.ToTensor(),
                transforms.CenterCrop(size_B),
                transforms.Normalize((0.5), (0.5))
                ]
test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B)

netG_A2B.train()
netG_B2A.train()
netD_A.train()
netD_B.train()


# tensorboard
# train_log_dir = opt.log_dir_train + opt.log
# test_log_dir = opt.log_dir_train + opt.log
# train_summary_writer = summary.create_file_writer(train_log_dir)
# test_summary_writer = summary.create_file_writer(test_log_dir)
# iteration = 0


# logger = logging.getLogger('mylogger')
# logger.setLevel(logging.DEBUG)

# timestamp = str(int(time.time()))
# fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
# fh.setLevel(logging.DEBUG)
 
# # 再创建一个handler，用于输出到控制台
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)


###### Training ######
end = time.time()
for epoch in range(epoch, n_epochs):
    # writer = SummaryWriter(train_log_dir)
    real_out, fake_out = None, None

    for idx, batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ######### (1) forward #########

        ## G A->B##
        hf = utils.Gaussian_pass(real_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = (hf+real_A)/2.0
        lf = utils.Gaussian_pass(real_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
        lf_feature_A, hf_feature_A, fake_B = netG_A2B(lf, hf) # A2B: lf_feature, hf_feature, rc

        # tv_fake_B = criterion_ssim_TV_loss(fake_B)*0.5
        ## idt A->A ##
        _, _, idt_A = netG_B2A(hf, lf)


        hf_feature_A = hf_feature_A.detach()
        hf_feature_A.requires_grad = False
        lf_feature_A = lf_feature_A.detach()
        lf_feature_A.requires_grad = False

        ## G B->A ##
        hf = utils.Gaussian_pass(fake_B[0], high_pass=True, band=opt.h2l_hb).unsqueeze(0).unsqueeze(0)
        hf = (hf+fake_B)/2.0
        lf = utils.Gaussian_pass(fake_B[0], high_pass=False, band=opt.h2l_lb).unsqueeze(0).unsqueeze(0)
        # lf = (lf+fake_B)/2.0
        hf_feature_recovered_A, lf_feature_recovered_A, recovered_A = netG_B2A(hf, lf) # B2A: hf_feature, lf_feature, rc
        

        ## G B->A ##
        hf = utils.Gaussian_pass(real_B[0], high_pass=True, band=opt.h2l_hb).unsqueeze(0).unsqueeze(0)
        hf = (hf+real_B)/2.0
        lf = utils.Gaussian_pass(real_B[0], high_pass=False, band=opt.h2l_lb).unsqueeze(0).unsqueeze(0)
        hf_feature_B, lf_feature_B, fake_A = netG_B2A(hf, lf)

        ## idt B->B ##
        _, _, idt_B = netG_A2B(lf, hf)

        lf_feature_B = lf_feature_B.detach()
        lf_feature_B.requires_grad = False
        hf_feature_B = hf_feature_B.detach()
        hf_feature_B.requires_grad = False

        ## G A->B ##
        hf = utils.Gaussian_pass(fake_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = (hf+fake_A)/2.0
        lf = utils.Gaussian_pass(fake_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
        lf_feature_recovered_B, hf_feature_recovered_B, recovered_B = netG_A2B(lf, hf)


        ###### (2) G_A and G_B ######
        set_requires_grad([netD_A, netD_B], False)
        optimizer_G.zero_grad()

        wB_l, wB_h = utils.weighted_D(real_B[0], fake_B.detach()[0])
        pred_fake = netD_B(fake_B, wB_l, wB_h)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*opt.beta4

        wA_l, wA_h = utils.weighted_D(real_A[0], fake_A.detach()[0])
        pred_fake = netD_A(fake_A, wA_l, wA_h)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*opt.beta5



        ###### Loss function for generators ######
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.beta3 #+ opt.beta1*criterion_feature(lf_feature_A, lf_feature_recovered_A) 
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.beta3 #+ opt.beta1*criterion_feature(lf_feature_B, lf_feature_recovered_B) 
        loss_idt = criterion_identity(real_A, idt_A)*opt.beta2 +  criterion_identity(real_B, idt_B)*opt.beta2
        # loss_perceptual = criterion_perceptual.get_loss(recovered_A.repeat(1,3,1,1), real_A.repeat(1,3,1,1))
        # loss_ssim = (1- criterion_ssim(recovered_A, real_A)) + (1 - criterion_ssim(recovered_B, real_B) )

        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt
        loss_G.backward()
        optimizer_G.step()

        ###### (3) D_A and D_B ######
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D.zero_grad()

        ###### Loss function for discriminators ######
        # Real loss
        pred_real = netD_A(real_A, wA_l, wA_h)
        loss_D_real = criterion_GAN(pred_real, target_real)
        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach(), wA_l, wA_h)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()


        # Real loss
        pred_real = netD_B(real_B, wB_l, wB_h)
        loss_D_real = criterion_GAN(pred_real, target_real)      
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach(), wB_l, wB_h)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D.step()

        # if idx%10==0:
        #     iteration += 1
        #     writer.add_scalar('loss_idt', loss_idt, iteration)
        #     writer.add_scalar('loss_D_A', loss_D_A, iteration)
        #     writer.add_scalar('loss_D_B', loss_D_B, iteration)
        #     writer.add_scalar('loss_GAN_A2B', loss_GAN_A2B, iteration)
        #     writer.add_scalar('loss_GAN_B2A', loss_GAN_B2A, iteration)
        #     writer.add_scalar('loss_cycle_ABA', loss_cycle_ABA, iteration)
        #     writer.add_scalar('loss_cycle_BAB', loss_cycle_BAB, iteration)
        
        
        ####################################
        ####################################
        ###### training examples ######
        if idx == 1:
            input_tmp = Tensor(batchSize, input_nc, size_A, size_A)
            x = Variable(input_tmp.copy_(batch['A']))
            real_out = x
            hf = utils.Gaussian_pass(x[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            hf = (hf+x)/2.0
            lf = utils.Gaussian_pass(x[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            _, _, fake_out = netG_A2B(lf, hf)
      
    save_sample(epoch, real_out, "_input")
    save_sample(epoch, fake_out, "_output", Azimuthal=True)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # save checkpoints
    if opt.pretrained:
        if (epoch<opt.decay_epoch and epoch%5==4) or (epoch>=opt.decay_epoch):
            torch.save(netG_A2B.state_dict(), './checkpoint/netG_A2B_epoch'+str(epoch+1)+'.pth')
            if opt.B2A:
                torch.save(netG_B2A.state_dict(), './checkpoint/netG_B2A_epoch'+str(epoch+1)+'.pth')
    else:
        if epoch%3==2:
            torch.save(netG_A2B.state_dict(), './checkpoint/netG_A2B_epoch'+str(epoch+1)+'.pth')
            if opt.B2A:
                torch.save(netG_B2A.state_dict(), './checkpoint/netG_B2A_epoch'+str(epoch+1)+'.pth')
            
    print("Epoch (%d/%d) Finished" % (epoch+1, n_epochs))

    # evaluations
    # TODO: training loss, 3mm evaluation (test) set, and 6mm test set.

    mse = eval(netG_A2B, opt, epoch=epoch)
    mse_6mm = eval_6m(netG_A2B, test_dataset, opt)
    netG_A2B.train()
    # writer.add_scalar('MSE', mse, epoch+1)
    # writer.add_scalar('MSE_6mm', mse_6mm, epoch+1)
    # writer.close()
    
    train_time = time.time() - end
    end = time.time()
    print("---------- time cost: %d s----------"%(train_time))
