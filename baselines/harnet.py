import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
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
import itertools
import matplotlib.pyplot as plt
import skimage.metrics
from tqdm import tqdm
from azimuthal import FSIM, azimuthalAverage
from scipy.interpolate import griddata



def eval(model):
    lr = "../dataset/test/6mm_x2/"
    hr = "../dataset/test/3mm/"
    num, psnr, ssim, mse, nmi, fsim= 0, 0, 0, 0, 0, 0
    model.eval()
    T_1 = transforms.Compose([ transforms.ToTensor(),
                transforms.CenterCrop(size_A),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.2312), (0.1381)),
                 ])
    T_2 = transforms.Compose([ transforms.ToTensor(),
                transforms.CenterCrop(size_B),                         
                transforms.Normalize((0.5), (0.5)),
                # transforms.Normalize((0.2656), (0.1924)),
                ])
    for i in tqdm(range(297)):
        lr_path = os.path.join(lr, str(i)+"_6.png")
        hr_path = os.path.join(hr, str(i)+"_3.png")
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            lr_img = Image.open(lr_path).convert('L')
            hr_img = Image.open(hr_path).convert('L')

            lr_img = T_1(lr_img).cuda().unsqueeze(0)
            hr_img = T_2(hr_img).cuda().unsqueeze(0)

            sr_img = model(lr_img)

            yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg, data_range=2))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            fsim += FSIM(gtimg, yimg)
            num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f FSIM: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num, fsim/num))



class MyDateSet(Dataset):
    def __init__(self, imageFolder, labelFolder):
        self.imageFolder = imageFolder
        self.images = os.listdir(imageFolder)
        self.labelFolder = labelFolder
        self.labels = os.listdir(labelFolder)
        

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, index):
        lr = self.images[index]
        image_name = os.path.join(self.imageFolder, lr)
        hr = lr.replace("_6", "_3")
        # label_name = self.labels[index]
        label_name = os.path.join(self.labelFolder, hr)
        
        # image_read = cv2.imread(image_name)
        # label_read = cv2.imread(label_name)
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.CenterCrop(320),
                            transforms.Normalize((0.5), (0.5))
                        ])
        try:
            image_read = Image.open(image_name).convert('L')
            label_read = Image.open(label_name).convert('L')
        except:
            print(image_name)
            print(label_name)
        
        X = transform(image_read)
        Y = transform(label_read)

        return X, Y





class HARNet(nn.Module):
    def __init__(self):
        super(HARNet, self).__init__()
        self.conv_block = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        model1 = []
        for _ in range(20):
            model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block1 = nn.Sequential(*model1)
        model2 = []
        for _ in range(20):
            model2 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block2 = nn.Sequential(*model2)
        model3 = []
        for _ in range(20):
            model3 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block3 = nn.Sequential(*model3)
        model4 = []
        for _ in range(20):
            model4 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        self.conv_block4 = nn.Sequential(*model4)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(384, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        output = self.relu(self.conv1(x)) # (1, 128, 400, 400)

        block_residual = output # skip connection
        output = self.relu(self.conv2(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block1(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 192, 400, 400)

        block_residual = output
        output = self.relu(self.conv3(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block2(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 256, 400, 400)

        block_residual = output
        output = self.relu(self.conv4(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block3(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 320, 400, 400)

        block_residual = output
        output = self.relu(self.conv5(output)) # (1, 64, 400, 400)
        # for _ in range(19):
        #   output = self.relu(self.conv_block(output)) # (1, 64, 400, 400)
        output = self.relu(self.conv_block4(output)) 
        output = torch.cat((output, block_residual), 1) # (1, 384, 400, 400)

        output = self.conv6(output)

        output += x
        
        return output

class ImageDataset_6mm(Dataset):
    def __init__(self, root, transforms_A=None, transforms_B=None, mode='train'):
        self.transformA = transforms.Compose(transforms_A)
        self.transformB = transforms.Compose(transforms_B)


        self.files_A = sorted(glob.glob(os.path.join(root, 'LR') + '/*.*'))
        # self.files_B = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        path_A = self.files_A[index % len(self.files_A)]

        path_B = path_A
        path_B = path_B.replace("_lr.", "_hr.").replace("LR", "HR")

        item_A = self.transformA(Image.open(path_A).convert('L'))
        item_B = self.transformB(Image.open(path_B).convert('L'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return len(self.files_A)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)





def save_sample(epoch, tensor, suffix="_real", Azimuthal=False):
    output = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    f = np.fft.fft2(output)
    fshift = np.fft.fftshift(f)
    amp = np.log(np.abs(fshift))
    plt.imsave('./baseline_checkpoint/checkpoint_baseline_harnet/image_'+str(epoch+1)+suffix+'.jpeg', output, cmap="gray")
    plt.imsave('./baseline_checkpoint/checkpoint_baseline_harnet/image_'+str(epoch+1)+suffix+'amplitude.jpeg', amp, cmap="gray")
    if Azimuthal:
        N = 200
        epsilon = 1e-7
        fshift += epsilon

        magnitude_spectrum = 20*np.log(np.abs(fshift))

        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = azimuthalAverage(magnitude_spectrum)

        # Calculate the azimuthally averaged 1D power spectrum
        points = np.linspace(0,N,num=psd1D.size) # coordinates of a
        xi = np.linspace(0,N,num=N) # coordinates for interpolation
        lr = griddata(points,psd1D,xi,method='cubic')

        # interpolated = (interpolated-np.min(interpolated))/(np.max(interpolated)-np.min(interpolated))
        # psd1D_total[cont,:] = interpolated  
        plt.plot(xi, lr)
        plt.savefig('./baseline_checkpoint/checkpoint_baseline_harnet/image_'+str(epoch+1)+suffix+'azimuthal.jpeg')
        plt.cla()


# def save_checkpoint(model, epoch):
#     model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
#     state = {"epoch": epoch ,"model": model}
#     if not os.path.exists("./checkpoint/"):
#         os.makedirs("./checkpoint/")

#     torch.save(state, model_out_path)

#     print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    input_nc = 1
    output_nc = 1
    batchSize = 1
    size_A, size_B = 256, 256
    lr = 2e-4
    n_epochs, epoch, decay_epoch = 100, 0, 10
    n_cpu = 2

    data_set = MyDateSet("../dataset/train/paired/6mm", "../dataset/train/paired/3mm")
    train_data = DataLoader(dataset=data_set, num_workers=2, batch_size=1, shuffle=True)
    test_path = "../dataset/evalution_6mm/parts"
    transforms_A = [ 
                    transforms.ToTensor(),                 
                    transforms.CenterCrop(size_A),
                    transforms.Normalize((0.5), (0.5)),
                    # transforms.Resize((128, 128))
                    ]
    transforms_B = [ 
                    transforms.ToTensor(),
                    transforms.CenterCrop(size_B),
                    transforms.Normalize((0.5), (0.5))
                    ]
    test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B)

    cuda = torch.cuda.is_available()

    model = HARNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.apply(weights_init_normal)
    mse_loss = nn.MSELoss()
    adam = optim.Adam(model.parameters(), lr=1e3)

    
    print("baseline: HARNet")
    for epoch in range(100):
        print("epoch %d/100" %(epoch+1))
        X, out = None, None
        model.train()
        for i, data in enumerate(train_data):
            x, y = data
            
            # if (x!=torch.tensor(0)).item():
            X, Y = Variable(x), Variable(y)
            # print(X.data.size())
            if cuda:
                X = X.cuda()
                Y = Y.cuda()
            out = model(X)
            adam.zero_grad()
            loss = mse_loss(out, Y)
            loss.backward()
            adam.step()

        eval(model)
        eval_6m_baseline(model, test_dataset)

        
        save_sample(epoch, X, "_input")
        save_sample(epoch, out, "_output", Azimuthal=True)
        # save_checkpoint(model, epoch)
    print('Finished Training ')
