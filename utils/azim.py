import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import griddata

from .filter import Gaussian_pass
from .func import *



def plot_azimuthal(model,dataset, epoch,opt):
    n = len(dataset)
    model.eval()
    N = 128
    epsilon = 1e-7
    y = None
    xi = np.linspace(0,N,num=N) # coordinates for interpolation
    for i in range(n):
        img = dataset[i]['A'].unsqueeze(0).cuda()
        hf = Gaussian_pass(img[0], high_pass=True, band=opt.l2h_hb).unsqueeze(0).unsqueeze(0)
        hf = (hf+img)/2.0
        lf = Gaussian_pass(img[0], high_pass=False, band=opt.l2h_lb).unsqueeze(0).unsqueeze(0)
        _, _, yimg = model(lf, hf)
        yimg = yimg.cpu().detach().numpy().squeeze(0).squeeze(0)
        f = np.fft.fft2(yimg)
        fshift = np.fft.fftshift(f)
        fshift += epsilon
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(magnitude_spectrum)
        # Calculate the azimuthally averaged 1D power spectrum
        points = np.linspace(0,N,num=psd1D.size) # coordinates of a
        
        line = griddata(points,psd1D,xi,method='cubic')
        if y is None:
            y = line
        else:
            y = np.vstack((y,line))
    
    std = np.std(y, axis=0)
    ave = np.mean(y, axis=0)
    plt.plot(xi, ave, color='deeppink', label='frequency')
    # plt.plot(xi, lr, 'b-', label='ideal')
    plt.fill_between(xi, ave - std, ave + std, color='violet', alpha=0.2)
    plt.legend()
    plt.savefig('./results/image_ours_'+str(epoch+1)+'_azimuthal.jpeg')
    plt.cla()
    model.eval()

def ds_info(dataset):
    m_A, m_B, st_A, st_B = 0, 0, 0, 0
    for i in range(len(dataset)):
        m_A += torch.mean(dataset[i]['A'])
        m_B += torch.mean(dataset[i]['B'])
        st_A += torch.std(dataset[i]['A'])
        st_B += torch.std(dataset[i]['B'])
    print(m_A/len(dataset), m_B/len(dataset), st_A/len(dataset), st_B/len(dataset))

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