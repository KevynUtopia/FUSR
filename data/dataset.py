import glob
import random
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Dataloader
class ImageDataset(Dataset):
    def __init__(self, root, transforms_A=None, transforms_B=None, unaligned=True):
        self.transformA = transforms.Compose(transforms_A)
        self.transformB = transforms.Compose(transforms_B)

        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '6mm_x2') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '3mm') + '/*.*'))

    def __getitem__(self, index):
        ixd_A = index % len(self.files_A)
        img_A = Image.open(self.files_A[ixd_A]).convert('L')
        item_A = self.transformA(img_A)

        if self.unaligned:
            idx_B = random.randint(0, len(self.files_B) - 1)
            while idx_B==ixd_A:
                idx_B = random.randint(0, len(self.files_B) - 1)
            item_B = self.transformB(Image.open(self.files_B[idx_B]).convert('L'))
        else:
            item_B = self.transformB(Image.open(self.files_B[index % len(self.files_B)]).convert('L'))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

        

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
    
def create_dataset(opt):
    batchSize = opt.batchSize
    size_A, size_B = opt.sizeA, opt.sizeB
    dataroot = opt.dataroot

    transforms_A = [ 
                    transforms.ToTensor(),
                    transforms.RandomCrop((size_A, size_A)),
                    # transforms.Resize((size_A, size_A)),
                    transforms.Normalize((0.5), (0.5))                
                    ]
                    
    transforms_B = [ 
                    transforms.ToTensor(),
                    transforms.RandomCrop((size_B, size_B)),
                    transforms.Normalize((0.5), (0.5)),
                    ]

    dataset = ImageDataset(dataroot, transforms_A=transforms_A, transforms_B=transforms_B, unaligned=True)
    # print(len(dataset))


    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)


    ###################################
    # load (whole) 6mm eval set
    test_path = opt.whole_dir
    transforms_A = [ 
                    transforms.ToTensor(),                 
                    transforms.CenterCrop(size_A),
                    # transforms.Resize((size_A, size_A)),
                    transforms.Normalize((0.5), (0.5)),
                    # transforms.Normalize((0.280), (0.178)),
                    ]
    transforms_B = [ 
                    transforms.ToTensor(),
                    transforms.CenterCrop(size_B),
                    transforms.Normalize((0.5), (0.5)),
                    # transforms.Normalize((0.285), (0.214))
                    ]
    test_dataset = ImageDataset_6mm(test_path, transforms_A=transforms_A, transforms_B=transforms_B)

    return dataloader, test_dataset, len(dataset), len(test_dataset)
