import random
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from .filter import *
import logging


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
    # elif classname.find('ConvTranspose2d'):
    #     torch.nn.init.normal(m.weight.data, std=0.001)
    #     for name, _ in m.named_parameters():
    #         if name in ['bias']:
    #             torch.nn.init.constant_(m.bias.data, 0)

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
    # output = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    output = tensor2img(tensor)
    f = np.fft.fft2(output)
    fshift = np.fft.fftshift(f)
    amp = np.log(np.abs(fshift))
    plt.imsave('./results/image_ours_'+str(epoch+1)+suffix+'.jpeg', output, cmap="gray")
    plt.imsave('./results/image_ours_'+str(epoch+1)+suffix+'amplitude.jpeg', amp, cmap="gray")
    # if Azimuthal:
    #     N = 200
    #     epsilon = 1e-7
    #     fshift += epsilon

    #     magnitude_spectrum = 20*np.log(np.abs(fshift))

    #     # Calculate the azimuthally averaged 1D power spectrum
    #     psd1D = azimuthalAverage(magnitude_spectrum)

    #     # Calculate the azimuthally averaged 1D power spectrum
    #     points = np.linspace(0,N,num=psd1D.size) # coordinates of a
    #     xi = np.linspace(0,N,num=N) # coordinates for interpolation
    #     lr = griddata(points,psd1D,xi,method='cubic')

    #     # interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
    #     # psd1D_total[cont,:] = interpolated  
    #     plt.plot(xi, lr)
    #     plt.savefig('./results/image_ours_'+str(epoch+1)+suffix+'azimuthal.jpeg')
    #     plt.cla()




def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach()
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_checkpoint(model, epoch, opt, n_epochs, results_3m, results_6m):
    if not os.path.isdir(opt.check_dir):
        os.makedirs(opt.check_dir)
    target = opt.check_dir + '/netG_A2B_epoch'+str(epoch+1)+'.pth'
    results_3m_val, results_3m_test = results_3m
    results_6m_val, results_6m_test = results_6m
    results = {'epoch': epoch,
               'results_3m_val':results_3m_val,
               'results_3m_test':results_3m_test,
               'results_6m_val': results_6m_val,
               'results_6m_test': results_6m_test,
               'opt':opt,
                'net': model}
    torch.save(results, target)

def train_vis(netG_A2B, opt, img, Tensor, epoch,  **kwargs):
    netG_A2B.eval()
    input_tmp = Tensor(opt.batchSize, opt.input_nc, opt.sizeA, opt.sizeA)
    x = Variable(input_tmp.copy_(img))
    real_out = x
    hf = Gaussian_pass(x[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
    hf = opt.hfb*hf+ x
    lf = Gaussian_pass(x[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
    _, _, fake_out = netG_A2B(lf, hf)
    netG_A2B.train()
    if 'writer' in kwargs:
        if kwargs['writer'] != None:
            output = tensor2img(fake_out)
            kwargs['writer'].add_image('out_img', np.expand_dims(output, axis=0), global_step=epoch)

    save_sample(epoch, real_out, "_input")
    save_sample(epoch, fake_out, "_output", Azimuthal=True)

 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger



