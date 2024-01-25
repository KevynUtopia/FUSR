#!/usr/bin/python3


# import logging
import argparse

import time

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchinfo import summary
import itertools

import warnings
warnings.filterwarnings('ignore')

from utils import *
from metrics import *
from model import *
from data import *
from focal_frequency_loss import FocalFrequencyLoss as FFL
from args import *

# from tensorflow import summary
from tensorboardX import SummaryWriter


def main(opt):
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
    netD_A = FS_DiscriminatorA(opt.input_D_nc, n_layers=opt.n_layers)
    netD_B = FS_DiscriminatorB(opt.input_D_nc, n_layers=opt.n_layers)

    if cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # if not opt.pretrained:
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # summary(netG_A2B_enc)
    # summary(netD_A)

    # Lossess
    ffl = FFL(loss_weight=1.0, alpha=1.0)  
    ffl_l = FFL(loss_weight=1.0, alpha=1.0)  
    ffl_h = FFL(loss_weight=opt.loss_weight, alpha=opt.loss_alpha)  
    criterion_GAN = torch.nn.MSELoss()
    # criterion_cycle = torch.nn.SmoothL1Loss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_phase = phase_consistency_loss()
    criterion_identity = torch.nn.L1Loss()
    # criterion_perceptual = PerceptualLoss(torch.nn.MSELoss())
    criterion_ssim = SSIM()

    criterion_feature = torch.nn.BCEWithLogitsLoss()

    # Optimizers & LR schedulers
    optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=lr, betas=(0.9, 0.999))

    # if opt.scheduler:
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.9, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step, verbose=True)


    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_A = Tensor(batchSize, input_nc, size_A, size_A)
    input_B = Tensor(batchSize, output_nc, size_B, size_B)
    target_real = Variable(Tensor(opt.dis_size, opt.dis_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.dis_size, opt.dis_size).fill_(0.0), requires_grad=False)

    target_real = torch.flatten(target_real)
    target_fake = torch.flatten(target_fake)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    dataloader, test_dataset, trainset_len, _ = create_dataset(opt)

    # Set training log
    # TODO: add script to customize logger names (indicate experiment names)
    if not os.path.isdir(opt.log_dir_train):
        os.makedirs(opt.log_dir_train)
    if not os.path.isdir(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    if opt.tensorboard:
        writer = SummaryWriter(opt.tensorboard_dir, flush_secs=1)
    else:
        writer = None
    file = time.strftime("%m%d_%H%M", time.localtime())+'.log'
    logger = get_logger(opt.log_dir_train + file)
    logger.info('Start Training')


    ##### Training ######
    end = time.time()
    for epoch in range(0, n_epochs):

        # print("Epoch (%d/%d) Finished" % (epoch+1, n_epochs))
        logger.info('Epoch (%d/%d) Begins'%(epoch+1, n_epochs))

        for idx, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))


            ######### (1) forward #########

            ## G A->B##
            hf = Gaussian_pass(real_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ real_A
            lf = Gaussian_pass(real_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            lf_feature_A, hf_feature_A, fake_B = netG_A2B(lf, hf) # A2B: lf_feature, hf_feature, rc
            # tv_fake_B = criterion_ssim_TV_loss(fake_B)*0.5
            ## idt A->A ##
            _, _, idt_A = netG_B2A(hf, lf)

            hf_feature_A = hf_feature_A.detach()
            hf_feature_A.requires_grad = False
            lf_feature_A = lf_feature_A.detach()
            lf_feature_A.requires_grad = False

            ## G B->A ##
            hf = Gaussian_pass(fake_B[0], high_pass=True, band=opt.h2l_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ fake_B
            lf = Gaussian_pass(fake_B[0], high_pass=False, band=opt.h2l_lb).unsqueeze(0).unsqueeze(0)
            # lf = (lf+fake_B)/2.0
            hf_feature_recovered_A, lf_feature_recovered_A, recovered_A = netG_B2A(hf, lf) # B2A: hf_feature, lf_feature, rc

            ## G B->A ##
            hf = Gaussian_pass(real_B[0], high_pass=True, band=opt.h2l_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ real_B
            lf = Gaussian_pass(real_B[0], high_pass=False, band=opt.h2l_lb).unsqueeze(0).unsqueeze(0)
            hf_feature_B, lf_feature_B, fake_A = netG_B2A(hf, lf)

            ## idt B->B ##
            _, _, idt_B = netG_A2B(lf, hf)

            lf_feature_B = lf_feature_B.detach()
            lf_feature_B.requires_grad = False
            hf_feature_B = hf_feature_B.detach()
            hf_feature_B.requires_grad = False

            ## G A->B ##
            hf = Gaussian_pass(fake_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ fake_A
            lf = Gaussian_pass(fake_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            lf_feature_recovered_B, hf_feature_recovered_B, recovered_B = netG_A2B(lf, hf)


            ###### (2) G_A and G_B ######
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()

            wB_l, wB_h = weighted_D(real_B[0], fake_B.detach()[0])
            pred_fake = netD_B(fake_B, wB_l, wB_h, x_gt=real_A)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)*opt.beta5

            wA_l, wA_h = weighted_D(real_A[0], fake_A.detach()[0])
            pred_fake = netD_A(fake_A, wA_l, wA_h, x_gt=real_B)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)*opt.beta6



            ###### Loss function for generators ######
            recovered_A_hf = Gaussian_pass(recovered_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            recovered_A_lf = Gaussian_pass(recovered_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            real_A_hf = Gaussian_pass(real_A[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            real_A_lf = Gaussian_pass(real_A[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.beta1 + ffl_h(recovered_A_hf, real_A_hf)[0]*opt.beta7 + ffl_l(recovered_A_lf, real_A_lf)[0] #+ opt.beta1*criterion_feature(lf_feature_A, lf_feature_recovered_A) 
            
            recovered_B_hf = Gaussian_pass(recovered_B[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            recovered_B_lf = Gaussian_pass(recovered_B[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            real_B_hf = Gaussian_pass(real_B[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            real_B_lf = Gaussian_pass(real_B[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.beta2 + ffl_h(recovered_B_hf, real_B_hf)[0]*opt.beta7 + ffl_l(recovered_B_lf, real_B_lf)[0] #+ opt.beta1*criterion_feature(lf_feature_B, lf_feature_recovered_B) 
            loss_idt = criterion_identity(real_A, idt_A)*opt.beta3 +  criterion_identity(real_B, idt_B)*opt.beta4


            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_idt
            loss_G.backward()
            optimizer_G.step()

            ###### (3) D_A and D_B ######
            for _ in range(opt.dis_iter):
                set_requires_grad([netD_A, netD_B], True)
                optimizer_D.zero_grad()

                ###### Loss function for discriminators ######
                # Real loss
                pred_real = netD_A(real_A, wA_l, wA_h, x_gt=real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)
                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach(), wA_l, wA_h, x_gt=real_B)
                loss_D_fake = criterion_GAN(pred_fake, target_fake)
                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()


                # Real loss
                pred_real = netD_B(real_B, wB_l, wB_h, x_gt=real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)      
                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach(), wB_l, wB_h, x_gt=real_A)
                loss_D_fake = criterion_GAN(pred_fake, target_fake)
                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()
                optimizer_D.step()

            
            ###### visualize training examples ######
            if idx == 1:
                train_vis(netG_A2B, opt, batch['A'], Tensor, epoch, writer=writer if writer!= None else None)
            if writer!= None:
                writer.add_scalar('loss_GAN_A2B', loss_GAN_A2B.item(), epoch * trainset_len + idx)
                writer.add_scalar('loss_GAN_B2A', loss_GAN_B2A.item(), epoch * trainset_len + idx)
                writer.add_scalar('loss_cycle_ABA', loss_cycle_ABA.item(), epoch * trainset_len + idx)
                writer.add_scalar('loss_cycle_BAB', loss_cycle_BAB.item(), epoch * trainset_len + idx)
                writer.add_scalar('loss_D_A', loss_D_A.item(), epoch * trainset_len + idx)
                writer.add_scalar('loss_D_B', loss_D_B.item(), epoch * trainset_len + idx)


        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        results_3m, results_6m = model_evaluation(netG_A2B, opt, epoch, test_dataset, ffl, logger)
        # save checkpoints
        save_checkpoint(netG_A2B, epoch, opt, n_epochs, results_3m, results_6m)
        logger.info("Checkpoint Saved")
            

    
        train_time = time.time() - end
        end = time.time()
        logger.info('----------Epoch Time Cost: %d s----------'%(int(train_time)))
        # print("---------- time cost: %d s----------"%(train_time))

if __name__ == "__main__":
    opt = train_arguments()
    main(opt)
