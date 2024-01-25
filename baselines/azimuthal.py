import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


import phasepack.phasecong as pc
import cv2
from scipy.interpolate import griddata



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
