from tkinter import NE

from cv2 import namedWindow
from pytorch_wavelets import DWTForward
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append("..")
from utils import high_pass, low_pass


class phase_consistency_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        radius = 5
        rows, cols = x[0][0].shape
        center = int(rows / 2), int(cols / 2)

        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance_u_v = (i - center[0]) ** 2 + (j - center[1]) ** 2
                mask[i, j] = 1 - np.exp(-0.5 *  distance_u_v / (radius ** 2))
        m = torch.from_numpy(mask).float().cuda()

        f_x = torch.fft.fft2(x[0])
        fshift_x = torch.fft.fftshift(f_x)
        amp_x = (m * torch.log(torch.abs(fshift_x))).flatten()
        f_y = torch.fft.fft2(y[0])
        fshift_y = torch.fft.fftshift(f_y)
        amp_y = (m * torch.log(torch.abs(fshift_y))).flatten()
        return -torch.cosine_similarity(amp_x, amp_y, dim=0)

# Loss functions
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model
		
	def __init__(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()
			
	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss
