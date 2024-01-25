import glob
import cv2
from scipy.interpolate import griddata
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.metrics
import torchvision.transforms as transforms
import torch
# from metric import FSIM
from utils import tensor2img


transforms_B = [ transforms.ToTensor(),transforms.CenterCrop(256),transforms.Normalize((0.5), (0.5))]
transformB = transforms.Compose(transforms_B)
timg = transformB(Image.open("./dataset/eval/lr/lr_294.png").convert('L'))

band = 13
lf = transforms.GaussianBlur(band)(timg)
hf = torch.clamp((timg-lf), min=0.0)
# lf = torch.clamp((lf), min=0.0)

plt.imshow(lf.squeeze(), "gray")
plt.axis("off")

plt.imshow(0.5*hf.squeeze()+0.5*timg.squeeze(), "gray")
plt.axis("off")

plt.imshow(0.8*hf.squeeze()+0.2*timg.squeeze(), "gray")
plt.axis("off")

plt.imshow(1.0*hf.squeeze()+0.0*timg.squeeze(), "gray")
plt.axis("off")