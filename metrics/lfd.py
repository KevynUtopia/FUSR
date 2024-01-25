import numpy as np


def lfd(img1, img2):
    """Calculate LFD (Log Frequency Distance).

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LFD calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lfd result.
    """

    freq1 = np.fft.fft2(img1)
    freq2 = np.fft.fft2(img2)
    return np.log(np.mean((freq1.real - freq2.real)**2 + (freq1.imag - freq2.imag)**2) + 1.0)
