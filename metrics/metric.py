import random
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import skimage.metrics
from functools import lru_cache


import sys
sys.path.append("..")
from utils import Gaussian_pass, save_sample, tensor2img, plot_azimuthal

from .fsim import FSIM
from .lfd import lfd


###############################################################################################

@lru_cache()
def read_img(lr_path, hr_path, T_1, T_2):
    lr_img = Image.open(lr_path).convert('L')
    hr_img = Image.open(hr_path).convert('L')
    lr_img = T_1(lr_img).cuda().unsqueeze(0)
    hr_img = T_2(hr_img).cuda().unsqueeze(0)
    return lr_img, hr_img

def eval_visual(model, opt, epoch=0, ffl=None, ffl_h=False, ffl_l=False):
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
    i = random.randint(0, 296)


    lr_path = os.path.join(lr, str(i)+"_6.png")
    hr_path = os.path.join(hr, str(i)+"_3.png")

    ct = 0
    fmap_ffl, fmap_ffl_h, fmap_ffl_l = None, None, None
    for i in range(297):
        lr_path = os.path.join(lr, str(i)+"_6.png")
        hr_path = os.path.join(hr, str(i)+"_3.png")
        if not os.path.isfile(lr_path) or not os.path.isfile(hr_path):
            continue
        ct += 1
        lr_img = Image.open(lr_path).convert('L')
        hr_img = Image.open(hr_path).convert('L')
        
        lr_img = T_1(lr_img).cuda().unsqueeze(0)
        hr_img = T_2(hr_img).cuda().unsqueeze(0)
        
        hf = Gaussian_pass(lr_img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = opt.hfb*hf+ lr_img
        lf = Gaussian_pass(lr_img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)


        _, _, sr_img = model(lf, hf)

        if ffl is not None:
            if fmap_ffl is None:
                fmap_ffl = ffl(sr_img, lr_img)[1]
            else:
                fmap_ffl += ffl(sr_img, lr_img)[1]


            if ffl_h:
                hf_real = Gaussian_pass(hr_img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)

                if fmap_ffl_h is None:
                    fmap_ffl_h = ffl(hf_real, hf)[1]
                else:
                    fmap_ffl_h += ffl(hf_real, hf)[1]

            
            if ffl_l:
                lf_real = Gaussian_pass(hr_img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)

                if fmap_ffl_l is None:
                    fmap_ffl_l = ffl(lf_real, lf)[1]
                else:
                    fmap_ffl_l += ffl(lf_real, lf)[1]


    if fmap_ffl_h is not None:
        fmap_ffl = fmap_ffl/ct
        dir = "./ffl_np"
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = "ffl_"+str(epoch)+".npy"
        with open(os.path.join(dir, file), 'wb+') as f:
            np.save(f, fmap_ffl)

    if fmap_ffl_h is not None:
        fmap_ffl_h = fmap_ffl_h/ct
        dir = "./ffl_np"
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = "ffl_h_"+str(epoch)+".npy"
        with open(os.path.join(dir, file), 'wb+') as f:
            np.save(f, fmap_ffl_h)

    if fmap_ffl_h is not None:
        fmap_ffl_l = fmap_ffl_l/ct
        dir = "./ffl_np"
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = "ffl_l_"+str(epoch)+".npy"
        with open(os.path.join(dir, file), 'wb+') as f:
            np.save(f, fmap_ffl_l)


    save_sample(epoch, lr_img, "_eval_input")
    save_sample(epoch, sr_img, "_eval_output")


def eval(model, logger, opt, epoch=0):  
    lr_dir=  "./dataset/test/6mm_x2/"
    hr_pdir =  "./dataset/test/3mm/"
    lr = os.listdir(lr_dir)
    hr = os.listdir(hr_pdir)
    random.shuffle(lr)
    num, psnr, ssim, mse, nmi, fsim, lfdloss= 0, 0, 0, 0, 0, 0, 0
    model.eval()
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
    val_len = int(len(lr)*opt.p_val)
    for i in (range(val_len)):
        # lr_path = os.path.join(lr, str(i)+"_6.png")
        # hr_path = os.path.join(hr, str(i)+"_3.png")
        lr_path = os.path.join(lr_dir, lr[i])
        hr_path = lr_path.replace("_6", "_3").replace("6mm_x2", "3mm")
        # print(lr_path, hr_path, os.path.isfile(lr_path), os.path.isfile(hr_path))
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            
            lr_img, hr_img = read_img(lr_path, hr_path, T_1, T_2)
            
            hf = Gaussian_pass(lr_img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ lr_img
            lf = Gaussian_pass(lr_img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            _, _, sr_img = model(lf, hf)

            
            # yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            # gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            yimg = tensor2img(sr_img)
            gtimg = tensor2img(hr_img)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            fsim += FSIM(gtimg, yimg)
            lfdloss += lfd(gtimg, yimg)
            
            num += 1

    results_val = {"PSNR": "%.4f"%(psnr/num), "SSIM":"%.4f"%(ssim/num), "NMI":"%.4f"%(nmi/num), "FSIM":"%.4f"%(fsim/num), "LFD":"%.4f"%(lfdloss/num)}
    val_foveal = 'Vald 3mm: PSNR:{:.4f} SSIM:{:.4f} NMI:{:.4f} FSIM:{:.4f} LFD:{:.3f}'.format((psnr/num) , (ssim/num), (nmi/num), (fsim/num), (lfdloss/num))
    # logger.info(val_foveal)

    num, psnr, ssim, mse, nmi, fsim, lfdloss = 0, 0, 0, 0, 0, 0, 0
    for i in (range(val_len, len(lr))):
        # lr_path = os.path.join(lr, str(i)+"_6.png")
        # hr_path = os.path.join(hr, str(i)+"_3.png")
        lr_path = os.path.join(lr_dir, lr[i])
        hr_path = lr_path.replace("_6", "_3").replace("6mm_x2", "3mm")
        # print(lr_path, hr_path, os.path.isfile(lr_path), os.path.isfile(hr_path))
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            
            lr_img, hr_img = read_img(lr_path, hr_path, T_1, T_2)
            
            hf = Gaussian_pass(lr_img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
            hf = opt.hfb*hf+ lr_img
            lf = Gaussian_pass(lr_img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
            _, _, sr_img = model(lf, hf)

            
            # yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            # gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            yimg = tensor2img(sr_img)
            gtimg = tensor2img(hr_img)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            fsim += FSIM(gtimg, yimg)
            lfdloss += lfd(gtimg, yimg)
            
            num += 1

    results_test = {"PSNR": "%.4f"%(psnr/num), "SSIM":"%.4f"%(ssim/num), "NMI":"%.4f"%(nmi/num), "FSIM":"%.4f"%(fsim/num), "LFD":"%.4f"%(lfdloss/num)}
    test_foveal = 'Test 3mm: PSNR:{:.4f} SSIM:{:.4f} NMI:{:.4f} FSIM:{:.4f} LFD:{:.3f}'.format((psnr/num) , (ssim/num), (nmi/num), (fsim/num), (lfdloss/num))
    # logger.info(test_foveal)

    # for key,value in results.items():
    #     print('{key}: {value}'.format(key = key, value = value), end=" ")
    # print()
    
    eval_visual(model, opt, epoch=epoch)
    return (results_val, results_test), val_foveal, test_foveal
    


def eval_6m(model, dataset, logger, opt):
    n = len(dataset)
    num, psnr, ssim, mse, nmi, fsim, lfdloss= 0, 0, 0, 0, 0, 0, 0
    model.eval()
    sample_list = np.arange(n)
    np.random.shuffle(sample_list)
    val_len = int(n*opt.p_val)
    for i in range(val_len):
        img = dataset[sample_list[i]]['A'].unsqueeze(0).cuda()
        gt = dataset[sample_list[i]]['B'].unsqueeze(0).cuda()


        hf = Gaussian_pass(img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = opt.hfb*hf + img
        lf = Gaussian_pass(img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
        _, _, y = model(lf, hf)
        
        # print(gt.shape)
        # print("fdsafasd fsdaf")
        # yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
        # gtimg = gt.cpu().detach().numpy().squeeze(0).squeeze(0)
        yimg = tensor2img(y)
        gtimg = tensor2img(gt)
        psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
        ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
        mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
        nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
        fsim += FSIM(gtimg, yimg)
        lfdloss += lfd(gtimg, yimg)
        num += 1
    results_val = {"PSNR": "%.4f"%(psnr/num), "SSIM":"%.4f"%(ssim/num), "NMI":"%.4f"%(nmi/num), "FSIM":"%.4f"%(fsim/num), "LFD":"%.4f"%(lfdloss/num)}
    val_parafovea = 'Vald 6mm: PSNR:{:.4f} SSIM:{:.4f} NMI:{:.4f} FSIM:{:.4f} LFD:{:.3f}'.format((psnr/num) , (ssim/num), (nmi/num), (fsim/num), (lfdloss/num))
    # logger.info(val_parafovea)

    num, psnr, ssim, mse, nmi, fsim, lfdloss = 0, 0, 0, 0, 0, 0, 0
    for i in range(val_len, n):
        img = dataset[sample_list[i]]['A'].unsqueeze(0).cuda()
        gt = dataset[sample_list[i]]['B'].unsqueeze(0).cuda()

        hf = Gaussian_pass(img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = opt.hfb * hf + img
        lf = Gaussian_pass(img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
        _, _, y = model(lf, hf)

        # print(gt.shape)
        # print("fdsafasd fsdaf")
        # yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
        # gtimg = gt.cpu().detach().numpy().squeeze(0).squeeze(0)
        yimg = tensor2img(y)
        gtimg = tensor2img(gt)
        psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
        ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
        mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
        nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
        fsim += FSIM(gtimg, yimg)
        lfdloss += lfd(gtimg, yimg)
        num += 1
    results_test = {"PSNR": "%.4f" % (psnr / num), "SSIM": "%.4f" % (ssim / num), "NMI": "%.4f" % (nmi / num),
               "FSIM": "%.4f" % (fsim / num), "LFD": "%.4f" % (lfdloss / num)}
    test_parafovea = 'Test 6mm: PSNR:{:.4f} SSIM:{:.4f} NMI:{:.4f} FSIM:{:.4f} LFD:{:.3f}'.format((psnr / num), (ssim / num), (nmi / num),
                                                                           (fsim / num), (lfdloss / num))
    # logger.info(test_parafovea)
    # for key,value in results.items():
    #     print('{key}: {value}'.format(key = key, value = value), end=" ")
    # print()

    # print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f FSIM: %.4f LFD: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num, fsim/num, lfdloss/num))

    return (results_val, results_test), val_parafovea, test_parafovea

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
    


def train_eval(dataset, model, opt):
    i = random.randint(0, len(dataset) - 1)
    img = dataset[i]['A']
    x = img.unsqueeze(0).cuda()
    hf = Gaussian_pass(x[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
    hf = (hf+x)/2.0

    lf = Gaussian_pass(x[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
    _, _, y = model(lf, hf)

    yimg = y.cpu().detach().numpy().squeeze(0).squeeze(0)
    psnr = skimage.metrics.peak_signal_noise_ratio(yimg, img.squeeze(0).cpu().detach().numpy(), data_range=2)
    ssim = skimage.metrics.structural_similarity(yimg, img.squeeze(0).cpu().detach().numpy())
    nmi = skimage.metrics.mean_squared_error(yimg, img.squeeze(0).cpu().detach().numpy())
    fsim = FSIM(img.squeeze(0).cpu().detach().numpy(), yimg)
    print("traning PSNR: %.4f SSIM: %.4f NMI: %.4f FSIM: %.4f"%(psnr, ssim, nmi, fsim))


# @lru_cache()
def model_evaluation(netG_A2B, opt, epoch, test_dataset, ffl, logger):
    netG_A2B.eval()
    results_3m, val_3m ,test_3m = eval(netG_A2B, logger, opt, epoch=epoch)
    results_6m, val_6m ,test_6m = eval_6m(netG_A2B, test_dataset, logger, opt)
    logger.info(val_3m)
    logger.info(val_6m)
    logger.info(test_3m)
    logger.info(test_6m)
    eval_visual(netG_A2B, opt, epoch=epoch, ffl=ffl, ffl_h=opt.ffl_h, ffl_l=opt.ffl_l)
    if epoch%5==0:
            plot_azimuthal(netG_A2B, test_dataset, epoch,opt)
    return results_3m, results_6m
